#shift alt a
import os
import sys
# Lấy đường dẫn thư mục hiện tại
current_dir = os.path.dirname(os.path.abspath(__file__))

# Thêm thư mục cần import vào sys.path
# sys.path.append(os.path.join(current_dir, "01_W_CPU"))
# sys.path.append(os.path.join(current_dir, "02_W_CPU_SS"))
sys.path.append(os.path.join(current_dir, "SEG_01_W_CPU"))
# sys.path.append(os.path.join(current_dir, "03_W_GPU"))
# sys.path.append(os.path.join(current_dir, "04_W_GPU_SS"))

# import lanczos_cpu as codechuan 
# import lanczos_cpu_coo as codechuan 
# import app_new as codechuan 
import lanczos as codechuan 
# import lanczos_cpu_ss_coo as codechuan 


def kiemThuChayNhieuLanMain(solan):
    chuoi = input("Nhập vào tên file log: ")
    # folder_path = input("Nhập vào đường dẫn thư mục chứa ảnh: ")
    # Đường dẫn tương đối đến một tệp
    kichthuocthumuc = int(input("nhập vào kích thước thư mục muốn chọn (0 hoặc 10 hoặc 60 hoặc 100): "))
    if (kichthuocthumuc == 60) :
        relative_path = "image_data_60"
    if (kichthuocthumuc == 100) :
        relative_path = "image_data_100"
    if (kichthuocthumuc == 0) :
        relative_path = "resized_image_files"
    if (kichthuocthumuc == 1) :
        relative_path = "1 anh 50 test"

    # Chuyển đổi thành đường dẫn tuyệt đối
    absolute_path = os.path.abspath(relative_path)
    folder_path = absolute_path

    for i in range(solan):
        codechuan.kiemThuChayNhieuLan(i, chuoi, folder_path)  # Gọi hàm từ file code_chuan_log.py

if __name__ == "__main__":
    solan = int(input("Nhập số lần kiểm thử: "))
    kiemThuChayNhieuLanMain(solan)