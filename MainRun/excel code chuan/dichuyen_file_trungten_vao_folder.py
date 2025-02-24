import os
import shutil

def move_files_by_prefix():
    # Nhận chuỗi tiền tố từ người dùng
    prefix = input("Nhập tiền tố của tập tin: ").strip()
    
    if not prefix:
        print("Tiền tố không được để trống!")
        return
    
    # Lấy đường dẫn thư mục hiện tại
    current_dir = os.getcwd()
    target_dir = os.path.join(current_dir, f"{prefix}_files")
    
    # Tạo thư mục đích nếu chưa tồn tại
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # Duyệt qua tất cả các tập tin trong thư mục hiện tại
    for filename in os.listdir(current_dir):
        file_path = os.path.join(current_dir, filename)
        
        # Kiểm tra nếu là tập tin và có tên bắt đầu bằng tiền tố
        if os.path.isfile(file_path) and filename.startswith(prefix):
            shutil.move(file_path, os.path.join(target_dir, filename))
            print(f"Đã di chuyển: {filename}")
    
    print("Hoàn tất!")

if __name__ == "__main__":
    move_files_by_prefix()
