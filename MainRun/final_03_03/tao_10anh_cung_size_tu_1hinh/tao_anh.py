from PIL import Image
import os
from tkinter import Tk, filedialog

def select_image():
    root = Tk()
    root.withdraw()  # Ẩn cửa sổ chính
    file_path = filedialog.askopenfilename(title="Chọn một ảnh", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
    return file_path

def duplicate_image(image_path, num_copies=9):
    if not os.path.exists(image_path):
        print("Ảnh không tồn tại!")
        return
    
    image = Image.open(image_path)
    file_name, file_ext = os.path.splitext(image_path)
    
    output_folder = os.path.join(os.path.dirname(image_path), "duplicates")
    os.makedirs(output_folder, exist_ok=True)
    
    for i in range(1, num_copies + 1):
        output_path = os.path.join(output_folder, f"{os.path.basename(file_name)}_copy{i}{file_ext}")
        image.save(output_path)
        print(f"Đã tạo: {output_path}")

if __name__ == "__main__":
    img_path = select_image()
    if img_path:
        duplicate_image(img_path)
