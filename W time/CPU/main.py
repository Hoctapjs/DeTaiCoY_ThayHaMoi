import code_chuan_log as codechuan  # Import file chứa các hàm cần sử dụng
""" import code_chuan_log_songsong_cpu as codechuan  # Import file chứa các hàm cần sử dụng """
# import code_caitien_histogram as codechuan  # Import file chứa các hàm cần sử dụng
# import code_chuan_COO as codechuan  # Import file chứa các hàm cần sử dụng
""" import testCOO_measure as codechuan  # Import file chứa các hàm cần sử dụng """
# import code_chuan_lanczos as codechuan  # Import file chứa các hàm cần sử dụng
# import code_chuan_lanczos_v2 as codechuan  # Import file chứa các hàm cần sử dụng
# import code_chuan_lanczos_v2_QR as codechuan  # Import file chứa các hàm cần sử dụng
# import code_chuan_COO_His as codechuan  # Import file chứa các hàm cần sử dụng
# import code_chuan_coo_his_chuanhoa as codechuan  # Import file chứa các hàm cần sử dụng


def kiemThuChayNhieuLanMain(solan):
    chuoi = input("Nhập vào tên file log: ")
    for i in range(solan):
        codechuan.kiemThuChayNhieuLan(i, chuoi)  # Gọi hàm từ file code_chuan_log.py

if __name__ == "__main__":
    solan = int(input("Số lần muốn kiểm thử: "))
    kiemThuChayNhieuLanMain(solan)
