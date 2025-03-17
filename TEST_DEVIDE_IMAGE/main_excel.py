import os
import sys
import cv2
import numpy as np
import scipy.io as sio

# Lấy đường dẫn thư mục hiện tại
current_dir = os.path.dirname(os.path.abspath(__file__))

# Thêm thư mục cần import vào sys.path (giữ nguyên phần comment của bạn)
# sys.path.append(os.path.join(current_dir, "01_W_CPU"))
# sys.path.append(os.path.join(current_dir, "02_W_CPU_SS"))
# sys.path.append(os.path.join(current_dir, "03_W_GPU"))
# sys.path.append(os.path.join(current_dir, "04_W_GPU_SS"))
# sys.path.append(os.path.join(current_dir, "SEG_01_W_CPU"))
# sys.path.append(os.path.join(current_dir, "SEG_02_W_CPU_SS"))

# Import file main
import main as codechuan 

def split_image_and_groundtruth(image_path, mat_path, output_dir):
    # Tạo thư mục đầu ra nếu chưa tồn tại
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        print(f"Không thể đọc ảnh: {image_path}")
        return False

    # Kích thước ảnh
    height, width, _ = image.shape

    # Cắt ảnh theo chiều dọc
    left_half = image[:, :width // 2]
    right_half = image[:, width // 2:]

    # Tạo tên file đầu ra
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    left_image_path = os.path.join(output_dir, f"{base_name}_left.jpg")
    right_image_path = os.path.join(output_dir, f"{base_name}_right.jpg")

    # Lưu hai phần ảnh
    cv2.imwrite(left_image_path, left_half)
    cv2.imwrite(right_image_path, right_half)

    # Đọc dữ liệu ground truth
    if os.path.exists(mat_path):
        mat_data = sio.loadmat(mat_path)
        ground_truth = mat_data['groundTruth']

        # Kiểm tra kích thước của ground truth
        gt_height, gt_width = ground_truth.shape[:2]

        # Cắt ground truth
        left_gt = ground_truth[:, :gt_width // 2]
        right_gt = ground_truth[:, gt_width // 2:]

        # Lưu hai phần ground truth
        left_mat_path = os.path.join(output_dir, f"{base_name}_left.mat")
        right_mat_path = os.path.join(output_dir, f"{base_name}_right.mat")
        sio.savemat(left_mat_path, {'groundTruth': left_gt})
        sio.savemat(right_mat_path, {'groundTruth': right_gt})
        print(f"Đã cắt và lưu ảnh + groundtruth cho {base_name}")
    else:
        print(f"Không tìm thấy file .mat cho {base_name}")
    
    return True

def kiemThuChayNhieuLanMain(solan):
    chuoi = input("Nhập vào tên file log: ")
    kichthuocthumuc = int(input("Nhập vào kích thước thư mục muốn chọn (nhập 9 để test): "))
    
    # Xác định thư mục đầu vào
    if kichthuocthumuc == 60:
        relative_path = "image_data_60"
    elif kichthuocthumuc == 100:
        relative_path = "image_data_100"
    elif kichthuocthumuc == 0:
        relative_path = "resized_image_files"
    elif kichthuocthumuc == 1:
        relative_path = "1 anh 50 test"
    elif kichthuocthumuc == 9:
        relative_path = "image/train"
    else:
        print("Kích thước thư mục không hợp lệ!")
        return

    # Chuyển đổi thành đường dẫn tuyệt đối
    input_folder = os.path.abspath(relative_path)
    
    # Tạo thư mục đầu ra cho ảnh đã cắt
    output_folder = os.path.join(input_folder, "split_output")
    
    # Duyệt qua tất cả ảnh trong thư mục đầu vào
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(input_folder, filename)
            mat_path = os.path.join(input_folder, f"{os.path.splitext(filename)[0]}.mat")
            split_image_and_groundtruth(image_path, mat_path, output_folder)

    # Gọi hàm kiểm thử từ main.py với thư mục đầu ra
    for i in range(solan):
        codechuan.kiemThuChayNhieuLan(i, chuoi, output_folder)  # Gọi hàm từ file main.py

if __name__ == "__main__":
    solan = int(input("Nhập số lần kiểm thử: "))
    kiemThuChayNhieuLanMain(solan)