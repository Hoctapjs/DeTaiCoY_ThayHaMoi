import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image

# Tạo cửa sổ Tkinter và ẩn nó đi
root = tk.Tk()
root.withdraw()

# Hộp thoại chọn ảnh
image_path = filedialog.askopenfilename(
    title="Chọn ảnh",
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
if img.size != (481, 321):
    print("Ảnh không có kích thước 481x321!")
    exit()

# Định nghĩa kích thước từng phần
part_width = 241  # Một nửa chiều rộng (chẵn lẻ do trừ dư)
part_height = 161  # Một nửa chiều cao

# Lấy tên file gốc (không có phần mở rộng)
base_name, ext = os.path.splitext(os.path.basename(image_path))

# Chia ảnh thành 4 phần
parts = [
    (0, 0, part_width, part_height),  # Góc trên trái
    (part_width, 0, 481, part_height),  # Góc trên phải
    (0, part_height, part_width, 321),  # Góc dưới trái
    (part_width, part_height, 481, 321)  # Góc dưới phải
]

# Chọn thư mục để lưu ảnh
output_dir = filedialog.askdirectory(title="Chọn thư mục lưu ảnh")
if not output_dir:
    print("Không có thư mục nào được chọn!")
    exit()

# Cắt và lưu từng phần
for i, box in enumerate(parts, start=1):
    part_img = img.crop(box)
    part_img.save(os.path.join(output_dir, f"{base_name}_part{i}{ext}"))

print("Chia ảnh thành công!")
