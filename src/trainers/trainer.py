import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import os
import csv
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
        self.patience_counter = 0
        
        # Initialize training history
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

    def _create_optimizer(self):
        backbone_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'fc' in name:
                    head_params.append(param)
                else:
                    backbone_params.append(param)
        
        # Add weight decay for regularization
        weight_decay = self.config.get('weight_decay', 0.0)
        
        return optim.Adam([
            {'params': backbone_params, 'lr': self.config['backbone_lr'], 'weight_decay': weight_decay},
            {'params': head_params, 'lr': self.config['head_lr'], 'weight_decay': weight_decay}
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
        early_stopping_patience = self.config.get('early_stopping_patience', 0)
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        history_file = f'logs/training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)
            
            print(f'Training Loss: {train_loss:.4f}, Training Acc: {train_acc:.2f}%')
            print(f'real data Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.2f}%')
            
            # Save to history
            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save checkpoint if model improved
            improved = self.save_checkpoint(epoch, val_loss, val_acc)
            if improved:
                print(f'Validation Loss Improved: {val_loss:.4f}\n')
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                
            # Early stopping check
            if early_stopping_patience > 0 and self.patience_counter >= early_stopping_patience:
                print(f'Early stopping triggered after {early_stopping_patience} epochs without improvement')
                break
        
        # Save training history to CSV
        self.save_history(history_file)
        
        # Clean up monitoring resources
        self.monitor.close()
    
    def save_history(self, filepath):
        """Save training history to CSV file"""
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
            for i in range(len(self.history['epoch'])):
                writer.writerow([
                    self.history['epoch'][i],
                    self.history['train_loss'][i],
                    self.history['train_acc'][i],
                    self.history['val_loss'][i],
                    self.history['val_acc'][i]
                ])
        print(f'Training history saved to {filepath}')