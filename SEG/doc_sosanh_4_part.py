import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
import os
from scipy.optimize import linear_sum_assignment

def load_segmentation_file(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:  # <== CẬP NHẬT Ở ĐÂY
        lines = file.readlines()
    
    width = int(lines[4].split()[1])
    height = int(lines[5].split()[1])
    
    data_index = lines.index('data\n') + 1
    seg_data = lines[data_index:]
    
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
    if not file_obj:
        return None
    
    image = load_segmentation_file(file_obj.name)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='tab20', interpolation='nearest')
    plt.title(f'Segmented Image: {os.path.basename(file_obj.name)}')
    plt.colorbar()
    
    filename = os.path.basename(file_obj.name)
    img_path = f"output_{filename}.png"
    plt.savefig(img_path)
    plt.close()
    
    return img_path

def match_labels(seg1, seg2):
    unique_labels1 = np.unique(seg1)
    unique_labels2 = np.unique(seg2)
    
    cost_matrix = np.zeros((len(unique_labels1), len(unique_labels2)))
    for i, label1 in enumerate(unique_labels1):
        for j, label2 in enumerate(unique_labels2):
            cost_matrix[i, j] = -np.sum((seg1 == label1) & (seg2 == label2))
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    label_mapping = {unique_labels2[j]: unique_labels1[i] for i, j in zip(row_ind, col_ind)}
    
    seg2_mapped = np.vectorize(lambda x: label_mapping.get(x, 0))(seg2)
    return seg2_mapped

def dice_coefficient(seg1, seg2):
    intersection = np.sum((seg1 == seg2) & (seg1 > 0))
    total_pixels = np.sum(seg1 > 0) + np.sum(seg2 > 0)
    
    dice_score = (2. * intersection) / total_pixels if total_pixels > 0 else 1.0
    return dice_score * 100

def compare_segmentations(file1, file2):
    if not file1 or not file2:
        return None, None, "⚠ Vui lòng tải lên hai tệp SEG hợp lệ!"
    
    seg1 = load_segmentation_file(file1.name)
    seg2 = load_segmentation_file(file2.name)
    
    if seg1.shape != seg2.shape:
        return None, None, "⚠ Kích thước của hai ảnh phân đoạn không khớp!"
    
    seg2_mapped = match_labels(seg1, seg2)
    dice_score = dice_coefficient(seg1, seg2)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(seg2_mapped, cmap='tab20', interpolation='nearest')
    plt.title('Mapped Segmentation Image')
    plt.colorbar()
    img_path = "output_mapped.png"
    plt.savefig(img_path)
    plt.close()
    
    return display_segmented_image(file1), img_path, f"Dice Similarity Coefficient: {dice_score:.2f}%"

title = "Multi-SEG File Viewer & Comparison"
description = "Upload two SEG files for comparison and four additional SEG files for independent visualization."

def process_segmentation(file1, file2, file3, file4, file5, file6):
    result1, result2, dice_text = compare_segmentations(file1, file2)
    
    seg3 = display_segmented_image(file3)
    seg4 = display_segmented_image(file4)
    seg5 = display_segmented_image(file5)
    seg6 = display_segmented_image(file6)

    return result1, result2, dice_text, seg3, seg4, seg5, seg6

with gr.Blocks() as iface:
    gr.Markdown(f"## {title}")
    gr.Markdown(description)

    with gr.Row():
        file1 = gr.File(label="Upload SEG File 1 (Comparison)")
        file2 = gr.File(label="Upload SEG File 2 (Comparison)")

    with gr.Row():
        file3 = gr.File(label="Upload SEG File 3 (Part 1)")
        file4 = gr.File(label="Upload SEG File 4 (Part 2)")
    
    with gr.Row():
        file5 = gr.File(label="Upload SEG File 5 (Part 3)")
        file6 = gr.File(label="Upload SEG File 6 (Part 4)")
    
    process_btn = gr.Button("Process")
    
    with gr.Row():
        result1 = gr.Image(type="filepath")
        result2 = gr.Image(type="filepath")
    
    dice_text = gr.Text()
    
    with gr.Row():
        seg3 = gr.Image(type="filepath")
        seg4 = gr.Image(type="filepath")
    
    with gr.Row():
        seg5 = gr.Image(type="filepath")
        seg6 = gr.Image(type="filepath")

    process_btn.click(
        fn=process_segmentation,
        inputs=[file1, file2, file3, file4, file5, file6],
        outputs=[result1, result2, dice_text, seg3, seg4, seg5, seg6]
    )

if __name__ == "__main__":
    iface.launch()
