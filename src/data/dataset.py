from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, use_path_files=True):
        """
        Args:
            root_dir (string): Root directory containing the angle-based subdirectories
            transform (callable, optional): Optional transform to be applied on a sample
            use_path_files (bool): If True, expect 'paths.txt' in each class directory
                                 containing absolute paths to images
        """
        self.root_dir = root_dir
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        self.classes = ['0-30', '30-60', '60-90']
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):                
                if use_path_files:
                    path_file = os.path.join(class_dir, 'paths.txt')
                    if os.path.exists(path_file):
                        valid_paths = 0
                        with open(path_file, 'r') as f:
                            for line in f:
                                img_path = line.strip()
                                if img_path and os.path.exists(img_path):
                                    self.samples.append((img_path, self.class_to_idx[class_name]))
                                    valid_paths += 1
                        if valid_paths == 0:
                            print(f"Warning: No valid image paths found in {path_file}")
                    else:
                        print(f"Warning: paths.txt not found in {class_dir}")
                else:
                    for fname in os.listdir(class_dir):
                        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                            self.samples.append((
                                os.path.join(class_dir, fname),
                                self.class_to_idx[class_name]
                            ))

    def __len__(self):
        return len(self.samples)   
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder black image in case of error
            if self.transform:
                import torch
                placeholder = torch.zeros((3, 224, 224))
            else:
                placeholder = Image.new('RGB', (224, 224), 'black')
            return placeholder, label