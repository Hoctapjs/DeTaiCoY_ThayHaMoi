""" import code_chuan_log as codechuan  # Import file chứa các hàm cần sử dụng """
# import code_caitien_histogram as codechuan  # Import file chứa các hàm cần sử dụng
# import code_chuan_COO as codechuan  # Import file chứa các hàm cần sử dụng
# import code_chuan_lanczos as codechuan  # Import file chứa các hàm cần sử dụng
# import code_chuan_lanczos_v2 as codechuan  # Import file chứa các hàm cần sử dụng
# import code_chuan_lanczos_v2_QR as codechuan  # Import file chứa các hàm cần sử dụng
# import code_chuan_COO_His as codechuan  # Import file chứa các hàm cần sử dụng
# import code_chuan_coo_his_chuanhoa as codechuan  # Import file chứa các hàm cần sử dụng
import code_caitien_histogram_mul as codechuan  # Import file chứa các hàm cần sử dụng


def kiemThuChayNhieuLanMain(solan):
    chuoi = input("Nhập vào tên file log: ")
    # folder_path = input("Nhập vào đường dẫn thư mục chứa ảnh: ")
    folder_path = r"C:\Users\Lenovo\Downloads\DETAI_TTNT\DeTaiCoY_ThayHaMoi\CPU_log\image_data"

    for i in range(solan):
        codechuan.kiemThuChayNhieuLan(i, chuoi, folder_path)  # Gọi hàm từ file code_chuan_log.py

if __name__ == "__main__":
    solan = int(input("Nhập số lần kiểm thử: "))
    kiemThuChayNhieuLanMain(solan)