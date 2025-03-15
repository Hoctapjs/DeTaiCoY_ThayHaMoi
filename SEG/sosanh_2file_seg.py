import tkinter as tk
from tkinter import filedialog
import numpy as np

def load_seg_file(file_path):
    """ Đọc file SEG và trả về ma trận nhãn """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Tìm phần dữ liệu bắt đầu từ 'data'
    data_index = lines.index("data\n") + 1
    header_lines = lines[:data_index - 1]
    data_lines = lines[data_index:]
    
    # Lấy kích thước ảnh từ header
    width = height = 0
    for line in header_lines:
        if line.startswith("width"):
            width = int(line.split()[1])
        elif line.startswith("height"):
            height = int(line.split()[1])
    
    labels = np.zeros((height, width), dtype=np.int32)
    
    # Parse dữ liệu pixel
    for line in data_lines:
        label, row, start_col, end_col = map(int, line.split())
        labels[row, start_col:end_col] = label
    
    return labels

def select_files():
    """ Hiển thị hộp thoại chọn 2 file SEG """
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(title="Chọn 2 file SEG", filetypes=[("SEG Files", "*.seg")])
    return file_paths if len(file_paths) == 2 else None

def compute_iou(seg1, seg2):
    intersection = np.sum((seg1 == seg2) & (seg1 > 0))
    union = np.sum((seg1 > 0) | (seg2 > 0))
    return intersection / union if union > 0 else 1.0

def main():
    file_paths = select_files()
    if not file_paths:
        print("❌ Cần chọn đúng 2 file SEG!")
        return
    
    seg1 = load_seg_file(file_paths[0])
    seg2 = load_seg_file(file_paths[1])
    
    iou = compute_iou(seg1, seg2)

    if iou == 1:
        print("Giống hoàn toàn")
        
    if iou < 1:
        print(f"Giống tương đối : {iou*100:.4f}%")

    if iou == 0:
        print("Khác hoàn toàn")
    
if __name__ == "__main__":
    main()
