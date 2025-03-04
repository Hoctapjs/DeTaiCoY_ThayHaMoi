from PIL import Image
import os

def resize_image(image_path, max_size=100):
    with Image.open(image_path) as img:
        img.thumbnail((max_size, max_size))  # Giữ nguyên tỷ lệ, không vượt quá max_size
        img.save(image_path)  # Ghi đè lên file cũ

def resize_images_in_directory():
    current_dir = os.getcwd()  # Lấy đường dẫn thư mục hiện tại
    
    for filename in os.listdir(current_dir):
        if filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
            image_path = os.path.join(current_dir, filename)
            resize_image(image_path)
            print(f"Resized: {filename}")

if __name__ == "__main__":
    resize_images_in_directory()
