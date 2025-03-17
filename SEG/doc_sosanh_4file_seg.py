import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
import os
from scipy.optimize import linear_sum_assignment

# Hàm đọc file .seg (giữ nguyên)
def load_segmentation_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    width = int(lines[4].split()[1])  # width value
    height = int(lines[5].split()[1])  # height value
    
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

# Hàm hiển thị ảnh phân đoạn (giữ nguyên)
def display_segmented_image(file_obj, output_name="output"):
    image = load_segmentation_file(file_obj.name)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='tab20', interpolation='nearest')
    plt.title(f'Restored Segmented Image: {os.path.basename(file_obj.name)}')
    plt.colorbar()
    
    img_path = f"{output_name}_{os.path.basename(file_obj.name)}.png"
    plt.savefig(img_path)
    plt.close()
    
    return img_path

# Hàm khớp nhãn giữa hai phân đoạn (giữ nguyên)
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

# Hàm tính Dice Coefficient (giữ nguyên)
def dice_coefficient(seg1, seg2):
    intersection = np.sum((seg1 == seg2) & (seg1 > 0))
    total_pixels = np.sum(seg1 > 0) + np.sum(seg2 > 0)
    
    dice_score = (2. * intersection) / total_pixels if total_pixels > 0 else 1.0
    return dice_score * 100

# Hàm so sánh 4 file .seg
def compare_segmentations(file1, file2, file3, file4):
    files = [file1, file2, file3, file4]
    valid_files = [f for f in files if f is not None]
    
    if len(valid_files) < 2:
        return [None] * 4, "⚠ Vui lòng tải lên ít nhất 2 tệp SEG hợp lệ để so sánh!"
    
    # Đọc các file .seg
    segmentations = [load_segmentation_file(f.name) for f in valid_files]
    file_names = [os.path.basename(f.name) for f in valid_files]
    
    # Kiểm tra kích thước
    shapes = [seg.shape for seg in segmentations]
    if len(set(shapes)) > 1:
        return [None] * 4, "⚠ Kích thước của các ảnh phân đoạn không khớp!"
    
    # Hiển thị từng ảnh phân đoạn
    output_images = []
    for i, file_obj in enumerate(valid_files):
        img_path = display_segmented_image(file_obj, output_name=f"output_part_{i+1}")
        output_images.append(img_path)
    
    # So sánh từng cặp (so với file đầu tiên làm chuẩn)
    dice_scores = []
    for i in range(1, len(segmentations)):
        seg_mapped = match_labels(segmentations[0], segmentations[i])
        dice_score = dice_coefficient(segmentations[0], seg_mapped)
        dice_scores.append(f"Dice Coefficient ({file_names[0]} vs {file_names[i]}): {dice_score:.2f}%")
    
    # Điền thêm None cho các vị trí trống trong output_images nếu thiếu file
    while len(output_images) < 4:
        output_images.append(None)
    
    result_text = "\n".join(dice_scores) if dice_scores else "Không có đủ file để so sánh."
    return output_images, result_text

# Cấu hình giao diện Gradio
title = "4 SEG Files Viewer & Comparison"
description = "Upload up to 4 .seg files to visualize and compare segmentation accuracy using the Dice coefficient (compared to the first file)."

iface = gr.Interface(
    fn=compare_segmentations,
    inputs=[
        gr.File(label="Upload SEG File 1 (Reference)"),
        gr.File(label="Upload SEG File 2"),
        gr.File(label="Upload SEG File 3"),
        gr.File(label="Upload SEG File 4")
    ],
    outputs=[
        gr.Gallery(type="filepath", label="Segmented Images", columns=2),
        gr.Text(label="Dice Coefficients")
    ],
    title=title,
    description=description
)

if __name__ == "__main__":
    iface.launch()