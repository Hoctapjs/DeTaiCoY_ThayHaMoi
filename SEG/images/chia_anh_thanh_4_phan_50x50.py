import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image

# Tạo cửa sổ Tkinter và ẩn nó đi
root = tk.Tk()
root.withdraw()

# Hộp thoại chọn ảnh
image_path = filedialog.askopenfilename(
    title="Chọn ảnh (50x50)",
    filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
)

if not image_path:
    print("Không có ảnh nào được chọn!")
    exit()

# Mở ảnh
try:
    img = Image.open(image_path)
except Exception as e:
    print(f"Lỗi khi mở ảnh: {e}")
    exit()

# Kiểm tra kích thước ảnh
if img.size != (50, 50):
    print("Ảnh không có kích thước 50x50!")
    exit()

# Định nghĩa kích thước từng phần
part_width, part_height = 25, 25  # Chia làm 4 phần

# Lấy tên file gốc
base_name, ext = os.path.splitext(os.path.basename(image_path))

# Chọn thư mục lưu ảnh
output_dir = filedialog.askdirectory(title="Chọn thư mục lưu ảnh")
if not output_dir:
    print("Không có thư mục nào được chọn!")
    exit()

# Cắt ảnh và lưu
parts = [
    (0, 0, part_width, part_height),  # Trên trái
    (part_width, 0, 50, part_height),  # Trên phải
    (0, part_height, part_width, 50),  # Dưới trái
    (part_width, part_height, 50, 50)  # Dưới phải
]

for i, box in enumerate(parts, start=1):
    part_img = img.crop(box)
    part_img.save(os.path.join(output_dir, f"{base_name}_part{i}{ext}"))

print("Chia ảnh 50x50 thành công!")
