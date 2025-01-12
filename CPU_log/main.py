import code_chuan_log as codechuan  # Import file chứa các hàm cần sử dụng

def kiemThuChayNhieuLanMain(solan):
    chuoi = input("Nhập vào tên file log: ")
    for i in range(solan):
        codechuan.kiemThuChayNhieuLan(i, chuoi)  # Gọi hàm từ file code_chuan_log.py

if __name__ == "__main__":
    solan = int(input("Số lần muốn kiểm thử: "))
    kiemThuChayNhieuLanMain(solan)
