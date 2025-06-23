import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import os
from src.utils.monitoring.training_monitor import TrainingMonitor

class ModelTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        synthetic_loader,
        config,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.synthetic_loader = synthetic_loader
        self.device = device
        self.config = config
        
        # Initialize optimizer with different learning rates
        self.optimizer = self._create_optimizer()
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=2, factor=0.5
        )
        
        # Initialize monitoring
        self.classes = ['0-30°', '30-60°', '60-90°']
        self.monitor = TrainingMonitor(
            model=self.model,
            classes=self.classes,
            enable_monitoring=config.get('enable_monitoring', True)
        )
        self.best_val_loss = float('inf')

    def _create_optimizer(self):
        backbone_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'fc' in name:
                    head_params.append(param)
                else:
                    backbone_params.append(param)
        
        return optim.Adam([
            {'params': backbone_params, 'lr': self.config['backbone_lr']},
            {'params': head_params, 'lr': self.config['head_lr']}
        ])

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_predicted = []
        print(f'Train loader length: {len(self.train_loader)}')
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            # Log training step
            self.monitor.log_training_step(
                model=self.model,
                optimizer=self.optimizer,
                images=images,
                labels=labels,
                outputs=outputs,
                loss=loss
            )
            
            # Update running statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Collect predictions for confusion matrix
            all_labels.extend(labels.cpu().numpy())
            all_predicted.extend(predicted.cpu().numpy())
            
            if batch_idx % self.config['log_interval'] == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(images)}/{len(self.train_loader.dataset)} '
                      f'({100. * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        train_loss = running_loss / len(self.train_loader)
        train_acc = 100. * correct / total
        
        return train_loss, train_acc

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        all_labels = []
        all_predicted = []
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                val_loss += self.criterion(outputs, labels).item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_labels.extend(labels.cpu().numpy())
                all_predicted.extend(predicted.cpu().numpy())
        
        val_loss /= len(self.val_loader)
        val_acc = 100. * correct / total
        
        # do the same for synthetic data
        synthetic_loss = 0
        synthetic_correct = 0
        synthetic_total = 0
        synthetic_all_labels = []
        synthetic_all_predicted = []

        with torch.no_grad():
            for images, labels in self.synthetic_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                synthetic_loss += self.criterion(outputs, labels).item()

                _, predicted = torch.max(outputs, 1)
                synthetic_total += labels.size(0)
                synthetic_correct += (predicted == labels).sum().item()

                synthetic_all_labels.extend(labels.cpu().numpy())
                synthetic_all_predicted.extend(predicted.cpu().numpy())

        synthetic_loss /= len(self.synthetic_loader)
        synthetic_acc = 100. * synthetic_correct / synthetic_total
        print(f'SYNTHETIC RESULTS - Epoch {epoch+1}')
        print(f'-SYNTHETIC- Loss: {synthetic_loss:.4f}, Acc: {synthetic_acc:.2f}%')

        # Log validation metrics
        self.monitor.log_validation_step(
            model=self.model,
            val_loss=val_loss,
            val_acc=val_acc,
            all_labels=all_labels,
            all_predictions=all_predicted
        )
        
        return val_loss, val_acc

    def save_checkpoint(self, epoch, val_loss, val_acc):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            checkpoint_dir = 'checkpoints'
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
                
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f'angle_classifier_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
            )
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }, checkpoint_path)
            
            print(f'Model saved to {checkpoint_path}')
            return True
        return False

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)
            
            print(f'Training Loss: {train_loss:.4f}, Training Acc: {train_acc:.2f}%')
            print(f'real data Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.2f}%')
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save checkpoint if model improved
            if self.save_checkpoint(epoch, val_loss, val_acc):
                print(f'Validation Loss Improved: {val_loss:.4f}\n')
        
        # Clean up monitoring resources
        self.monitor.close()