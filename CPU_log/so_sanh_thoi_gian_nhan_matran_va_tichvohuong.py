import re
import tkinter as tk
from tkinter import filedialog

def extract_times_from_file(file_path):
    """
    Đọc file log và tính trung bình thời gian từ các dòng log liên quan.
    """
    matrix_times = []
    dot_times = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Tìm thời gian của "Thoi gian nhan ma tran - vector (song song)"
            match_matrix = re.search(r"Thoi gian nhan ma tran - vector \(song song\): ([\d\.]+) giay", line)
            # match_matrix = re.search(r"Thoi gian nhan ma tran - vector: ([\d\.]+) giay", line)
            if match_matrix:
                matrix_times.append(float(match_matrix.group(1)))
            
            # Tìm thời gian của "Thoi gian tinh tich vo huong (song song)"
            match_dot = re.search(r"Thoi gian tinh tich vo huong \(song song\): ([\d\.]+) giay", line)
            # match_dot = re.search(r"Thoi gian tinh tich vo huong: ([\d\.]+) giay", line)
            if match_dot:
                dot_times.append(float(match_dot.group(1)))

    # Tính trung bình
    avg_matrix_time = sum(matrix_times) / len(matrix_times) if matrix_times else 0
    avg_dot_time = sum(dot_times) / len(dot_times) if dot_times else 0

    print(f"\n📂 File: {file_path}")
    print(f"✅ Trung bình thời gian nhân ma trận - vector: {avg_matrix_time:.6f} giây")
    print(f"✅ Trung bình thời gian tích vô hướng: {avg_dot_time:.6f} giây")

def select_file_and_process():
    """
    Hiển thị hộp thoại chọn file và xử lý file log được chọn.
    """
    root = tk.Tk()
    root.withdraw()  # Ẩn cửa sổ chính của Tkinter
    file_path = filedialog.askopenfilename(title="Chọn file log", filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
    
    if file_path:
        extract_times_from_file(file_path)
    else:
        print("❌ Không có file nào được chọn.")

if __name__ == "__main__":
    select_file_and_process()
