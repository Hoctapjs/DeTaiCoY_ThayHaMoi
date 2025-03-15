import numpy as np
from scipy.optimize import linear_sum_assignment
from tkinter import Tk, filedialog

def load_seg_file(file_path):
    """ Đọc file SEG và trả về ma trận nhãn phân đoạn """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    data_index = None
    width = height = 0
    
    for i, line in enumerate(lines):
        if "data" in line.lower():
            data_index = i + 1
            break
        elif line.startswith("width"):
            width = int(line.split()[1])
        elif line.startswith("height"):
            height = int(line.split()[1])

    if data_index is None:
        raise ValueError("⚠ Không tìm thấy phần 'data' trong file SEG!")

    labels = np.zeros((height, width), dtype=np.int32)

    for line in lines[data_index:]:
        values = line.split()
        if len(values) != 4:
            continue
        
        label, row, start_col, end_col = map(int, values)
        labels[row, start_col:end_col] = label

    return labels

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

# Mở hộp thoại chọn file
Tk().withdraw()
file_path_1 = filedialog.askopenfilename(title="Chọn file SEG thứ nhất")
file_path_2 = filedialog.askopenfilename(title="Chọn file SEG thứ hai")

# Kiểm tra nếu file được chọn
if not file_path_1 or not file_path_2:
    raise ValueError("⚠ Bạn phải chọn đủ hai file SEG!")

seg1 = load_seg_file(file_path_1)
seg2 = load_seg_file(file_path_2)

if seg1.shape != seg2.shape:
    raise ValueError("⚠ Kích thước của hai ảnh phân đoạn không khớp!")

seg2_mapped = match_labels(seg1, seg2)
dice_score = dice_coefficient(seg1, seg2)

print(f"Dice Similarity Coefficient: {dice_score:.2f}%")