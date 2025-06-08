import torch
import yaml
import torchvision.transforms as transforms
from PIL import Image
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
from src.models.angle_classifier import AngleClassifier

class AnglePredictor:
    def __init__(self, checkpoint_path=None):
        # Load configuration
        with open('configs/config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Set up device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = AngleClassifier(
            pretrained=False,  # We'll load weights from checkpoint
            freeze_backbone=True
        ).to(self.device)
        
        # Load latest checkpoint if none specified
        if checkpoint_path is None:
            checkpoints = [f for f in os.listdir('checkpoints') if f.endswith('.pth')]
            if not checkpoints:
                raise RuntimeError("No checkpoints found! Please train the model first.")
            checkpoint_path = os.path.join('checkpoints', max(checkpoints, key=lambda x: os.path.getmtime(os.path.join('checkpoints', x))))
            print(f"Loading latest checkpoint: {checkpoint_path}")
        
        # Load model weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Initialize transforms
        self.transform = transforms.Compose([
            transforms.Resize(self.config['data']['img_size']),
            transforms.CenterCrop(self.config['data']['img_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Class names
        self.classes = ['0-30°', '30-60°', '60-90°']

    def predict_image(self, image_path):
        """Predict camera angle for a single image"""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            'class': self.classes[predicted_class],
            'confidence': confidence * 100,
            'probabilities': {
                cls: prob.item() * 100 
                for cls, prob in zip(self.classes, probabilities[0])
            }
        }

    def predict_batch(self, image_dir):
        """Predict camera angles for all images in a directory"""
        results = {}
        image_extensions = ('.jpg', '.jpeg', '.png', '.webp')
        
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(image_extensions):
                    image_path = os.path.join(root, file)
                    try:
                        results[file] = self.predict_image(image_path)
                    except Exception as e:
                        results[file] = {'error': str(e)}
        
        return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Predict camera viewing angles from images')
    parser.add_argument('path', help='Path to an image or directory of images')
    parser.add_argument('--checkpoint', help='Path to specific model checkpoint (optional)')
    args = parser.parse_args()
    
    predictor = AnglePredictor(args.checkpoint)
    path = Path(args.path)
    
    if path.is_file():
        # Single image prediction
        result = predictor.predict_image(str(path))
        print(f"\nPredictions for {path.name}:")
        print(f"Predicted class: {result['class']}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print("\nClass probabilities:")
        for cls, prob in result['probabilities'].items():
            print(f"{cls}: {prob:.2f}%")
    
    else:
        # Batch prediction
        results = predictor.predict_batch(str(path))
        print(f"\nProcessed {len(results)} images:")
        
        # Group by predicted class
        by_class = {}
        for filename, result in results.items():
            if 'error' in result:
                print(f"\nError processing {filename}: {result['error']}")
                continue
                
            pred_class = result['class']
            if pred_class not in by_class:
                by_class[pred_class] = []
            by_class[pred_class].append((filename, result['confidence']))
        
        # Print summary
        for cls in predictor.classes:
            if cls in by_class:
                files = by_class[cls]
                print(f"\n{cls}: {len(files)} images")
                print("Top 5 most confident predictions:")
                for filename, conf in sorted(files, key=lambda x: x[1], reverse=True)[:5]:
                    print(f"  {filename}: {conf:.2f}% confidence")

if __name__ == '__main__':
    main()