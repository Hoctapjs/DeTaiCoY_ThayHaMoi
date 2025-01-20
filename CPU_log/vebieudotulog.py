import matplotlib.pyplot as plt
import re

# Hàm đọc log và trích xuất thời gian
def extract_times_from_log(log_file):
    times = []  # Danh sách để lưu thời gian
    with open(log_file, 'r') as file:
        for line in file:
            # Tìm kiếm thời gian trong dòng log
            match = re.search(r"Thoi gian: (\d+\.\d+) giay", line)
            if match:
                times.append(float(match.group(1)))  # Thêm thời gian vào danh sách
    return times

# Tên file log
log_file = '20_1_2025_Lanczos0.txt'

# Trích xuất thời gian từ file log
times = extract_times_from_log(log_file)

# In kết quả
print("Danh sách thời gian từ log:")
print(times)

# Vẽ biểu đồ cột theo dữ liệu từ mảng `times`
plt.figure(figsize=(10, 6))

# Vẽ cột
bar_positions = range(len(times))  # Tạo vị trí cho các cột
plt.bar(bar_positions, times, color='skyblue', width=0.6)

# Thêm nhãn giá trị trên các cột
for i, time in enumerate(times):
    plt.text(i, time + 0.1, f'{time:.2f}', ha='center', va='bottom')

# Thêm tiêu đề và nhãn
plt.title('Thời gian xử lý từ file log')
plt.xlabel('Lần chạy')
plt.ylabel('Thời gian (giây)')
plt.xticks(bar_positions, [f'Run {i+1}' for i in range(len(times))])  # Đặt nhãn trục X là các lần chạy

# Hiển thị biểu đồ
plt.tight_layout()
plt.show()
