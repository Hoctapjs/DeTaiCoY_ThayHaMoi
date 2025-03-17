import tkinter as tk
from tkinter import filedialog
from PIL import Image

# Tạo cửa sổ Tkinter và ẩn nó đi
root = tk.Tk()
root.withdraw()

# Chọn file hình ảnh
file_path = filedialog.askopenfilename(
    title="Chọn hình ảnh",
    filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
)

if not file_path:
    print("Không có hình ảnh nào được chọn!")
    exit()

try:
    # Nhập kích thước mới từ người dùng
    new_width = int(input("Nhập chiều rộng mới: "))
    new_height = int(input("Nhập chiều cao mới: "))

    # Mở ảnh và resize
    img = Image.open(file_path)
    resized_img = img.resize((new_width, new_height))


    # Chọn đường dẫn để lưu ảnh đã resize
    save_path = filedialog.asksaveasfilename(
        title="Lưu hình ảnh đã resize",
        defaultextension=".png",
        filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg"), ("All Files", "*.*")]
    )

    if save_path:
        resized_img.save(save_path)
        print(f"Hình ảnh đã được lưu tại: {save_path}")
    else:
        print("Không lưu được hình ảnh!")
except ValueError:
    print("Vui lòng nhập số nguyên hợp lệ!")
except Exception as e:
    print(f"Đã có lỗi xảy ra: {e}")
