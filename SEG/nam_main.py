import os
import sys
from skimage import io
import numpy as np

# Lấy đường dẫn thư mục hiện tại
current_dir = os.path.dirname(os.path.abspath(__file__))

# Thêm thư mục SEG_NAM vào sys.path
sys.path.append(os.path.join(current_dir, "SEG_NAM"))
import lanczosnam as codechuan

# Hàm cắt ảnh 200x200 thành 4 ảnh 50x50 và lưu vào folder
def split_and_save_image(image_path, output_folder):
    # Đọc ảnh
    image = io.imread(image_path)
    if image.shape[0] != 200 or image.shape[1] != 200:
        print(f"❌ Ảnh {image_path} không có kích thước 200x200!")
        return None
    
    # Tạo folder tạm nếu chưa tồn tại
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Cắt ảnh thành 4 phần 50x50
    h, w, c = image.shape  # h=200, w=200
    parts = [
        image[0:50, 0:50, :],         # Top-left (0-49, 0-49)
        image[0:50, 150:200, :],      # Top-right (0-49, 150-199)
        image[150:200, 0:50, :],      # Bottom-left (150-199, 0-49)
        image[150:200, 150:200, :]    # Bottom-right (150-199, 150-199)
    ]
    part_names = ["top_left", "top_right", "bottom_left", "bottom_right"]

    # Lưu từng phần vào folder tạm
    saved_paths = []
    imagename = os.path.splitext(os.path.basename(image_path))[0]
    for part, part_name in zip(parts, part_names):
        output_path = os.path.join(output_folder, f"{imagename}_{part_name}.png")
        io.imsave(output_path, part)
        saved_paths.append(output_path)
        print(f"✅ Đã lưu: {output_path}")
    
    return saved_paths

def kiemThuChayNhieuLanMain(solan):
    chuoi = input("Nhập vào tên file log: ")

    # Đường dẫn tới thư mục namimg
    relative_path = "namimg"
    absolute_path = os.path.abspath(relative_path)
    folder_path = absolute_path

    # Kiểm tra thư mục namimg
    if not os.path.isdir(folder_path):
        print(f"❌ Thư mục {folder_path} không tồn tại!")
        return

    # Lấy danh sách ảnh trong namimg
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    if not image_files:
        print(f"❌ Không tìm thấy file ảnh nào trong {folder_path}!")
        return

    # Chọn ảnh đầu tiên trong thư mục (hoặc bạn có thể thêm logic để chọn ảnh cụ thể)
    selected_image = os.path.join(folder_path, image_files[0])
    print(f"📷 Đã chọn ảnh: {selected_image}")

    # Tạo folder tạm để lưu 4 ảnh 50x50
    temp_folder = os.path.join(current_dir, "temp_split_images")
    
    # Cắt ảnh và lưu vào folder tạm
    split_image_paths = split_and_save_image(selected_image, temp_folder)
    if not split_image_paths:
        return

    # Chạy kiểm thử trên folder tạm
    for i in range(solan):
        print(f"🔄 Đang chạy lần kiểm thử {i+1}/{solan} trên folder tạm...")
        codechuan.kiemThuChayNhieuLan(i, chuoi, temp_folder)

    # (Tùy chọn) Xóa folder tạm sau khi hoàn tất
    # import shutil
    # shutil.rmtree(temp_folder)
    # print(f"🗑️ Đã xóa folder tạm: {temp_folder}")

if __name__ == "__main__":
    solan = int(input("Nhập số lần kiểm thử: "))
    kiemThuChayNhieuLanMain(solan)