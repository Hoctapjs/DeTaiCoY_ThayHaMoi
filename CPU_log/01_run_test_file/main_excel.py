#shift alt a
import os
import sys
# Lấy đường dẫn thư mục hiện tại
current_dir = os.path.dirname(os.path.abspath(__file__))

# Thêm thư mục cần import vào sys.path
# sys.path.append(os.path.join(current_dir, "01_lanczos_tri_rieng_CPU"))
# sys.path.append(os.path.join(current_dir, "02_lanczos_tri_rieng_CPU_COO"))
# sys.path.append(os.path.join(current_dir, "03_lanczos_tri_rieng_CPU_SS"))
sys.path.append(os.path.join(current_dir, "04_lanczos_tri_rieng_CPU_COO_SS"))

# import lanczos_cpu as codechuan 
# import lanczos_cpu_coo as codechuan 
import lanczos_cpu_ss_coo as codechuan 
# import lanczos_cpu as codechuan 


def kiemThuChayNhieuLanMain(solan):
    chuoi = input("Nhập vào tên file log: ")
    # folder_path = input("Nhập vào đường dẫn thư mục chứa ảnh: ")
    # Đường dẫn tương đối đến một tệp
    kichthuocthumuc = int(input("nhập vào kích thước thư mục muốn chọn (0 hoặc 60 hoặc 100): "))
    if (kichthuocthumuc == 60) :
        relative_path = "image_data_60"
    if (kichthuocthumuc == 100) :
        relative_path = "image_data_100"
    if (kichthuocthumuc == 0) :
        relative_path = "image"

    # Chuyển đổi thành đường dẫn tuyệt đối
    absolute_path = os.path.abspath(relative_path)
    folder_path = absolute_path

    for i in range(solan):
        codechuan.kiemThuChayNhieuLan(i, chuoi, folder_path)  # Gọi hàm từ file code_chuan_log.py

if __name__ == "__main__":
    solan = int(input("Nhập số lần kiểm thử: "))
    kiemThuChayNhieuLanMain(solan)