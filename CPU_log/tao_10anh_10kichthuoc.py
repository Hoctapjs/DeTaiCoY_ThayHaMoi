from PIL import Image
import tkinter as tk
from tkinter import filedialog

def resize_images():
    # Mở hộp thoại chọn file
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")])
    
    if not file_path:
        print("Không có ảnh nào được chọn!")
        return
    
    # Mở ảnh
    img = Image.open(file_path)
    
    # Tạo và lưu 10 ảnh với kích thước tăng dần
    for i in range(10):
        size = 50 + i * 5  # Kích thước ảnh
        resized_img = img.resize((size, size))
        save_path = f"resized_image_{size}x{size}.png"
        resized_img.save(save_path)
        print(f"Đã lưu ảnh: {save_path}")

    print("Hoàn thành!")

if __name__ == "__main__":
    resize_images()