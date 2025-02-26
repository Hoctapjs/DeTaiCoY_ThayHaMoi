import pandas as pd
import glob
import tkinter as tk
from tkinter import filedialog

# Mở hộp thoại chọn thư mục
root = tk.Tk()
root.withdraw()  # Ẩn cửa sổ chính
folder_path = filedialog.askdirectory(title="Chọn thư mục chứa file Excel")

if folder_path:  # Nếu người dùng chọn thư mục
    # Lấy danh sách tất cả file Excel trong thư mục
    file_list = glob.glob(f"{folder_path}/*.xlsx")

    # Tạo danh sách để chứa dữ liệu từ các file
    dataframes = []
    empty_row = pd.DataFrame([[]])  # Tạo 1 hàng trống

    # Đọc từng file và thêm vào danh sách
    for file in file_list:
        df = pd.read_excel(file)  # Đọc file Excel
        dataframes.append(df)     # Thêm dữ liệu
        dataframes.append(empty_row)  # Chèn hàng trống

    # Gộp tất cả dữ liệu thành một DataFrame
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Xuất ra file Excel mới trong thư mục đã chọn
    output_file = f"{folder_path}/merged_file.xlsx"
    merged_df.to_excel(output_file, index=False, header=True)

    print(f"Đã ghép xong! File kết quả: {output_file}")
else:
    print("Bạn chưa chọn thư mục nào!")
