import os
import tkinter as tk
from tkinter import filedialog, messagebox

def read_seg_file(file_path):
    """Đọc file .seg và tách phần metadata và data"""
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    metadata = []
    data = []
    data_start = False

    for line in lines:
        line = line.strip()
        if not data_start:
            metadata.append(line)
            if line == "data":
                data_start = True
        else:
            data.append(line)

    return metadata, data

def split_data(data, part):
    """Chia phần data phù hợp với từng phần ảnh"""
    x_start, x_end, y_start, y_end = part
    new_data = []

    for line in data:
        parts = list(map(int, line.split()))
        label, row, col_start, col_end = parts

        if y_start <= row <= y_end:  # Chỉ lấy các dòng trong phạm vi của part
            new_col_start = max(col_start, x_start)
            new_col_end = min(col_end, x_end)

            if new_col_start <= new_col_end:
                new_data.append(f"{label} {row - y_start} {new_col_start - x_start} {new_col_end - x_start}")

    return new_data

def save_seg_file(metadata, data, output_path, part_name):
    """Lưu file .seg cho từng phần"""
    new_metadata = []
    for line in metadata:
        if line.startswith("width"):
            new_metadata.append("width 25")
        elif line.startswith("height"):
            new_metadata.append("height 25")
        else:
            new_metadata.append(line)

    output_file = os.path.join(output_path, f"{part_name}.seg")
    with open(output_file, "w", encoding="utf-8") as file:
        file.write("\n".join(new_metadata) + "\n")
        file.write("\n".join(data) + "\n")
    
    print(f"Đã tạo: {output_file}")

def split_seg_file(input_file, output_folder):
    """Hàm chính để chia file .seg"""
    metadata, data = read_seg_file(input_file)

    # Xác định các phần của ảnh
    parts = {
        "top_left": (0, 24, 0, 24),
        "top_right": (25, 49, 0, 24),
        "bottom_left": (0, 24, 25, 49),
        "bottom_right": (25, 49, 25, 49)
    }

    for part_name, part_coords in parts.items():
        new_data = split_data(data, part_coords)
        save_seg_file(metadata, new_data, output_folder, part_name)

    messagebox.showinfo("Hoàn tất", "Đã chia file thành 4 phần!")

def select_file():
    """Hộp thoại chọn file .seg"""
    file_path = filedialog.askopenfilename(filetypes=[("SEG files", "*.seg")])
    if file_path:
        entry_file.delete(0, tk.END)
        entry_file.insert(0, file_path)

def select_folder():
    """Hộp thoại chọn thư mục lưu file"""
    folder_path = filedialog.askdirectory()
    if folder_path:
        entry_folder.delete(0, tk.END)
        entry_folder.insert(0, folder_path)

def process_seg_file():
    """Xử lý khi nhấn nút Chia File"""
    input_file = entry_file.get()
    output_folder = entry_folder.get()

    if not os.path.isfile(input_file):
        messagebox.showerror("Lỗi", "Vui lòng chọn file .seg hợp lệ!")
        return

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    split_seg_file(input_file, output_folder)

# Giao diện Tkinter
root = tk.Tk()
root.title("Chia file SEG thành 4 phần")
root.geometry("500x250")

tk.Label(root, text="Chọn file SEG nguồn:").pack(pady=5)
frame1 = tk.Frame(root)
frame1.pack()
entry_file = tk.Entry(frame1, width=50)
entry_file.pack(side=tk.LEFT, padx=5)
tk.Button(frame1, text="Chọn...", command=select_file).pack(side=tk.LEFT)

tk.Label(root, text="Chọn thư mục lưu file:").pack(pady=5)
frame2 = tk.Frame(root)
frame2.pack()
entry_folder = tk.Entry(frame2, width=50)
entry_folder.pack(side=tk.LEFT, padx=5)
tk.Button(frame2, text="Chọn...", command=select_folder).pack(side=tk.LEFT)

tk.Button(root, text="Chia File", command=process_seg_file, height=2, width=20, bg="green", fg="white").pack(pady=20)

root.mainloop()
