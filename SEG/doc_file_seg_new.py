import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

def load_segmentation_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Extract width and height from metadata
    width = int(lines[4].split()[1])  # width value
    height = int(lines[5].split()[1])  # height value
    
    # Find the start of data section
    data_index = lines.index('data\n') + 1
    seg_data = lines[data_index:]
    
    return width, height, seg_data

def restore_segmented_image(file_path):
    width, height, seg_data = load_segmentation_file(file_path)
    
    # Initialize the image with zeros (background)
    segmented_image = np.zeros((height, width), dtype=int)
    
    # Process the segmentation data
    for line in seg_data:
        parts = line.split()
        if len(parts) != 4:
            continue  # Skip invalid lines
        
        label = int(parts[0])
        row = int(parts[1])
        col_start = int(parts[2])
        col_end = int(parts[3])
        
        # Set the corresponding pixel values in the segmented image
        segmented_image[row, col_start:col_end+1] = label
    
    return segmented_image

def display_segmented_image(image):
    plt.imshow(image, cmap='tab20', interpolation='nearest')
    plt.title('Restored Segmented Image')
    plt.colorbar()
    plt.show()

def select_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select SEG File", filetypes=[("SEG Files", "*.seg"), ("All Files", "*.*")])
    return file_path

# Usage example
if __name__ == "__main__":
    file_path = select_file()
    if file_path:
        segmented_image = restore_segmented_image(file_path)
        display_segmented_image(segmented_image)
    else:
        print("No file selected.")
