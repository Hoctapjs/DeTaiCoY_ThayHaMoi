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

# Lấy kích thước ảnh gốc
img_width, img_height = img.size

# Định nghĩa kích thước từng phần (chia 2)
part_width = img_width // 2
part_height = img_height // 2

# Lấy tên file gốc (không có phần mở rộng)
base_name, ext = os.path.splitext(os.path.basename(image_path))

# Chia ảnh thành 4 phần
parts = [
    (0, 0, part_width, part_height),  # Góc trên trái
    (part_width, 0, img_width, part_height),  # Góc trên phải
    (0, part_height, part_width, img_height),  # Góc dưới trái
    (part_width, part_height, img_width, img_height)  # Góc dưới phải
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
