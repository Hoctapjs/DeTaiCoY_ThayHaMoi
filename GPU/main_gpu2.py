
import code_caitien_gpu_histogram_mul as codechuan  # Import file chứa các hàm cần sử dụng
import os

# Đường dẫn tương đối đến một tệp
relative_path = "GPU\image_data_60"

# Chuyển đổi thành đường dẫn tuyệt đối
absolute_path = os.path.abspath(relative_path)

def kiemThuChayNhieuLanMain(solan):
    chuoi = input("Nhập vào tên file log: ")
    # folder_path = input("Nhập vào đường dẫn thư mục chứa ảnh: ")
    # Đường dẫn tương đối đến một tệp
    relative_path = "image_data_60"

    # Chuyển đổi thành đường dẫn tuyệt đối
    absolute_path = os.path.abspath(relative_path)
    folder_path = absolute_path

    for i in range(solan):
        codechuan.kiemThuChayNhieuLan(i, chuoi, folder_path)  # Gọi hàm từ file code_chuan_log.py

if __name__ == "__main__":
    solan = int(input("Nhập số lần kiểm thử: "))
    kiemThuChayNhieuLanMain(solan)