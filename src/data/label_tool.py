import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import os
import shutil
from pathlib import Path

class ImageLabelTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera Angle Labeling Tool")
        
        # Initialize variables
        self.current_image_path = None
        self.image_list = []
        self.current_index = 0
        
        # Initialize splits and categories first
        self.splits = ['train', 'val', 'test']
        self.categories = ['0-30', '30-60', '60-90']
        
        # Create output directories
        self.create_output_directories()
        
        # Create main containers after initializing splits and categories
        self.create_gui_elements()
    
    def create_gui_elements(self):
        # Create frames
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        image_frame = ttk.Frame(self.root)
        image_frame.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)
        
        button_frame = ttk.Frame(self.root)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        
        # Control elements
        ttk.Button(control_frame, text="Load Directory", 
                  command=self.load_directory).pack(side=tk.LEFT, padx=5)
        
        # Image display
        self.image_label = ttk.Label(image_frame)
        self.image_label.pack(expand=True, fill=tk.BOTH)
        
        # Split selection
        split_frame = ttk.LabelFrame(button_frame, text="Dataset Split")
        split_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.split_var = tk.StringVar(value='train')
        for split in self.splits:
            ttk.Radiobutton(split_frame, text=split, value=split,
                          variable=self.split_var).pack(side=tk.LEFT, padx=5)
        
        # Angle buttons
        angle_frame = ttk.LabelFrame(button_frame, text="Camera Angle")
        angle_frame.pack(fill=tk.X, padx=5, pady=5)
        
        for category in self.categories:
            ttk.Button(angle_frame, text=f"{category}°",
                      command=lambda c=category: self.label_image(c)).pack(
                          side=tk.LEFT, expand=True, padx=5)
        
        # Navigation buttons
        nav_frame = ttk.Frame(button_frame)
        nav_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(nav_frame, text="Previous", 
                  command=self.prev_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Next",
                  command=self.next_image).pack(side=tk.RIGHT, padx=5)
        
        # Progress label
        self.progress_var = tk.StringVar(value="No images loaded")
        ttk.Label(nav_frame, textvariable=self.progress_var).pack(
            side=tk.LEFT, expand=True)
    
    def create_output_directories(self):
        base_dir = Path("data")
        for split in self.splits:
            for category in self.categories:
                os.makedirs(base_dir / split / category, exist_ok=True)
    
    def load_directory(self):
        dir_path = filedialog.askdirectory(title="Select Image Directory")
        if not dir_path:
            return
            
        self.image_list = [
            os.path.join(dir_path, f) for f in os.listdir(dir_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
        ]
        
        if self.image_list:
            self.current_index = 0
            self.show_current_image()
        else:
            self.progress_var.set("No images found in selected directory")
    
    def show_current_image(self):
        if not self.image_list:
            return
            
        self.current_image_path = self.image_list[self.current_index]
        
        # Load and resize image
        image = Image.open(self.current_image_path)
        
        # Calculate resize ratio while maintaining aspect ratio
        display_size = (800, 600)
        image.thumbnail(display_size, Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=photo)
        self.image_label.image = photo  # Keep a reference
        
        # Update progress
        self.progress_var.set(
            f"Image {self.current_index + 1} of {len(self.image_list)}")
    
    def label_image(self, category):
        if not self.current_image_path:
            return
            
        # Get destination path
        split = self.split_var.get()
        dest_dir = Path("data") / split / category
        
        # Copy image to appropriate directory
        filename = os.path.basename(self.current_image_path)
        dest_path = dest_dir / filename
        
        try:
            shutil.copy2(self.current_image_path, dest_path)
            print(f"Saved to {dest_path}")
            self.next_image()
        except Exception as e:
            print(f"Error saving image: {e}")
    
    def next_image(self):
        if self.image_list and self.current_index < len(self.image_list) - 1:
            self.current_index += 1
            self.show_current_image()
    
    def prev_image(self):
        if self.image_list and self.current_index > 0:
            self.current_index -= 1
            self.show_current_image()

def main():
    root = tk.Tk()
    app = ImageLabelTool(root)
    root.mainloop()

if __name__ == "__main__":
    main()