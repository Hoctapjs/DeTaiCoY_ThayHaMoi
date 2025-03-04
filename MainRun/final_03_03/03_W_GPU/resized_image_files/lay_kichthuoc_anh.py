from PIL import Image
import os

def get_image_sizes(image_path):
    with Image.open(image_path) as img:
        return img.size  # Trả về (width, height)

def log_image_sizes():
    current_dir = os.getcwd()  # Lấy đường dẫn thư mục hiện tại
    sizes = []
    log_file = os.path.join(current_dir, "image_sizes.log")
    
    with open(log_file, "w") as log:
        log.write(f"============\n")
        for filename in os.listdir(current_dir):
            if filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
                image_path = os.path.join(current_dir, filename)
                size = get_image_sizes(image_path)
                sizes.append((filename, size))
                log.write(f"{filename}: {size}\n")
                print(f"Logged: {filename} - {size}")
    
    return sizes

if __name__ == "__main__":
    image_sizes = log_image_sizes()
