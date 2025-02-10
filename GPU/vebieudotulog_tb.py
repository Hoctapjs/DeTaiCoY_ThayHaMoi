import matplotlib.pyplot as plt
import numpy as np
import re

# Hàm đọc log và trích xuất thời gian Tổng, COO và tên ảnh
def extract_data_from_log(log_file):
    image_names = []  # Danh sách tên ảnh
    total_times = []  # Danh sách thời gian tổng
    coo_times = []  # Danh sách thời gian COO
    with open(log_file, 'r', encoding='utf-8') as file:
        for line in file:
            # Tìm tên ảnh và thời gian
            match = re.search(r"(\d+\.jpg): ([\d\.]+) giây \(Tổng\) \| ([\d\.]+) giây \(COO\)", line)
            if match:
                image_names.append(match.group(1))  # Lấy tên ảnh
                total_times.append(float(match.group(2)))  # Lấy thời gian tổng
                coo_times.append(float(match.group(3)))  # Lấy thời gian COO
    return image_names, total_times, coo_times

# Tên file log
log_file = '10_2_2025_GPU_60x60_mul_summary.txt'

# Trích xuất dữ liệu từ file log
image_names, total_times, coo_times = extract_data_from_log(log_file)

# In kiểm tra
print("Tên ảnh:", image_names)
print("Thời gian Tổng:", total_times)
print("Thời gian COO:", coo_times)

# Xác định số lượng ảnh
n = len(total_times)
indices = np.arange(n)  # Vị trí cho các cột

# Độ rộng của mỗi cột (để có khoảng cách hợp lý)
width = 0.35

# Vẽ biểu đồ
plt.figure(figsize=(14, 6))
plt.bar(indices - width/2, total_times, width, label="Tổng", color='skyblue')
plt.bar(indices + width/2, coo_times, width, label="COO", color='salmon')

# Thêm nhãn giá trị trên các cột
for i in range(n):
    plt.text(indices[i] - width/2, total_times[i] + 0.05, f"{total_times[i]:.2f}", ha='center', va='bottom', fontsize=9)
    plt.text(indices[i] + width/2, coo_times[i] + 0.05, f"{coo_times[i]:.2f}", ha='center', va='bottom', fontsize=9)

# Cấu hình trục
plt.xticks(indices, image_names, rotation=45, ha="right")  # Hiển thị tên ảnh thay vì "Run X"
plt.ylabel("Thời gian (giây)")
plt.title("So sánh thời gian Tổng và COO theo từng ảnh")
plt.legend()  # Hiển thị chú thích

# Hiển thị biểu đồ
plt.tight_layout()
plt.show()
