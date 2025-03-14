import numpy as np
import matplotlib.pyplot as plt
import gradio as gr

def load_segmentation_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    width = int(lines[4].split()[1])  # width value
    height = int(lines[5].split()[1])  # height value
    
    data_index = lines.index('data\n') + 1
    seg_data = lines[data_index:]
    
    return width, height, seg_data

def restore_segmented_image(file_obj):
    file_path = file_obj.name  # Lấy đường dẫn tạm thời của file
    width, height, seg_data = load_segmentation_file(file_path)
    
    segmented_image = np.zeros((height, width), dtype=int)
    
    for line in seg_data:
        parts = line.split()
        if len(parts) != 4:
            continue  
        
        label = int(parts[0])
        row = int(parts[1])
        col_start = int(parts[2])
        col_end = int(parts[3])
        
        segmented_image[row, col_start:col_end+1] = label
    
    return segmented_image

def display_segmented_image(file_obj):
    image = restore_segmented_image(file_obj)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='tab20', interpolation='nearest')
    plt.title('Restored Segmented Image')
    plt.colorbar()
    
    img_path = "output.png"
    plt.savefig(img_path)
    plt.close()
    
    return img_path

iface = gr.Interface(
    fn=display_segmented_image,
    inputs=gr.File(label="Upload SEG File"),
    outputs=gr.Image(type="filepath"),
    title="SEG File Viewer",
    description="Upload a .seg file to visualize the segmented image."
)

if __name__ == "__main__":
    iface.launch()