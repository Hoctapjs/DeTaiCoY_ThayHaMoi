import matplotlib.pyplot as plt
import re

# hàm truyền vào số lượng cần chạy(5) + truyền tên file log chung (logapple3ver)
    # vòng lặp truyền tên chuẩn vào, cộng chuỗi i từ tên chuẩn
        # gọi file code_chuan_log (truyền tham số vào là tên chuẫn đã cộng số thứ tự)


# Hàm đọc log và trích xuất thời gian
def extract_times_from_log(log_file):
    times = []
    with open(log_file, 'r') as file:
        for line in file:
            match = re.search(r"Thoi gian: (\d+\.\d+) giay", line)  # Tìm kiếm thời gian trong dòng log
            if match:
                times.append(float(match.group(1)))  # Thêm thời gian vào danh sách
    return times

# Đọc thời gian từ hai file log
log1Name = 'logapple3.txt'
log2Name = 'logapple4.txt'
times_log1 = extract_times_from_log(log1Name)
times_log2 = extract_times_from_log(log2Name)

# Tính thời gian trung bình cho mỗi file log
avg_time_log1 = sum(times_log1) / len(times_log1) if times_log1 else 0
avg_time_log2 = sum(times_log2) / len(times_log2) if times_log2 else 0

# Vẽ biểu đồ cột
plt.figure(figsize=(10, 6))

# Vẽ biểu đồ cột cho log1 và log2
width = 0.35  # độ rộng của mỗi cột
bar1 = plt.bar([0], avg_time_log1, width=width * 0.5, label='Log 1', color='blue', align='center')  # Log 1
bar2 = plt.bar([1], avg_time_log2, width=width * 0.5, label='Log 2', color='red', align='center')  # Log 2

# Thêm giá trị vào đầu mỗi cột
plt.text(bar1[0].get_x() + bar1[0].get_width() / 2, avg_time_log1 + 0.1, f'{avg_time_log1:.2f}', ha='center', va='bottom')
plt.text(bar2[0].get_x() + bar2[0].get_width() / 2, avg_time_log2 + 0.1, f'{avg_time_log2:.2f}', ha='center', va='bottom')

# Thêm tiêu đề và nhãn
plt.title('So sánh thời gian xử lý giữa hai file log')
plt.xlabel('File log')
plt.ylabel('Thời gian trung bình (giây)')
plt.xticks([0, 1], [log1Name, log2Name])  # Đổi tên trục X thành tên file log
plt.legend()

# Hiển thị biểu đồ
plt.tight_layout()
plt.show()
