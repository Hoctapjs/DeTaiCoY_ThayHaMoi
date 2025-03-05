""" import code_chuan_log as codechuan  # Import file chứa các hàm cần sử dụng """
# import code_caitien_histogram as codechuan  # Import file chứa các hàm cần sử dụng
# import code_chuan_COO as codechuan  # Import file chứa các hàm cần sử dụng
# import code_chuan_lanczos as codechuan  # Import file chứa các hàm cần sử dụng
# import code_chuan_lanczos_v2 as codechuan  # Import file chứa các hàm cần sử dụng
# import code_chuan_lanczos_v2_original as codechuan  # Import file chứa các hàm cần sử dụng - giống bản v2 ở trên nhưng copy ra để test - thay thêm hàm kiemthuchaynhieulan với lại hàm normalized_cuts, save_segmentation và assign_labels
# import code_chuan_lanczos_v2_songsong as codechuan  # Import file chứa các hàm cần sử dụng
# import code_chuan_lanczos_v2_songsong_v2 as codechuan  # Import file chứa các hàm cần sử dụng
import code_chuan_lanczos_v2_songsong_v4 as codechuan  # Import file chứa các hàm cần sử dụng
# import code_chuan_lanczos_v2_QR as codechuan  # Import file chứa các hàm cần sử dụng
# import code_chuan_COO_His as codechuan  # Import file chứa các hàm cần sử dụng
# import code_chuan_coo_his_chuanhoa as codechuan  # Import file chứa các hàm cần sử dụng
# import code_caitien_histogram_mul as codechuan  # Import file chứa các hàm cần sử dụng
# import testCOO_measure as codechuan  # Import file chứa các hàm cần sử dụng

import os

# Đường dẫn tương đối đến một tệp
relative_path = "CPU_log\image_data_60"

# Chuyển đổi thành đường dẫn tuyệt đối
absolute_path = os.path.abspath(relative_path)

def kiemThuChayNhieuLanMain(solan):
    chuoi = input("Nhập vào tên file log: ")
    # folder_path = input("Nhập vào đường dẫn thư mục chứa ảnh: ")
    # Đường dẫn tương đối đến một tệp
    # relative_path = "resized_image_files_copy"
    relative_path = "img_large"
    # relative_path = "image_data_60"

    # Chuyển đổi thành đường dẫn tuyệt đối
    absolute_path = os.path.abspath(relative_path)  
    folder_path = absolute_path

    for i in range(solan):
        codechuan.kiemThuChayNhieuLan(i, chuoi, folder_path)  # Gọi hàm từ file code_chuan_log.py

if __name__ == "__main__":
    solan = int(input("Nhập số lần kiểm thử: "))
    kiemThuChayNhieuLanMain(solan)


    