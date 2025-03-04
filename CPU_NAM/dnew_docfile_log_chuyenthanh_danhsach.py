import re
import time
import tkinter as tk
from tkinter import filedialog
from collections import defaultdict
import matplotlib.pyplot as plt

def select_file():
    """Hiển thị hộp thoại để chọn file"""
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(title="Chọn file log", filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])

def parse_log_file(file_path):
    entries = []
    current_entry = {}
    start_time = time.time()  # Bắt đầu đo thời gian xử lý

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            match_file = re.search(r"file name: (\S+)", line)
            match_lan = re.search(r"Lan thu: (\d+)", line)
            match_time = re.search(r"Thoi gian: ([\d.]+)", line)

            if match_file:
                if current_entry:
                    entries.append(current_entry)
                current_entry = {"file_name": match_file.group(1)}
            elif match_lan:
                current_entry["lan_thu"] = int(match_lan.group(1))
            elif match_time:
                current_entry["thoi_gian"] = round(float(match_time.group(1)), 2)

        if current_entry:
            entries.append(current_entry)

    end_time = time.time()  # Kết thúc đo thời gian
    print(f"- INFO - Thoi gian xu ly (khong song song): {round(end_time - start_time, 2)} giây")
    return entries

def calculate_average_time(entries):
    time_data = defaultdict(lambda: {"total_time": 0, "count": 0})

    for entry in entries:
        file_name = entry["file_name"]
        time_data[file_name]["total_time"] += entry["thoi_gian"]
        time_data[file_name]["count"] += 1

    averages = {file: round(data["total_time"] / data["count"], 2) for file, data in time_data.items()}
    return averages

def calculate_overall_average(average_times):
    if not average_times:
        return 0
    return round(sum(average_times.values()) / len(average_times), 2)

def plot_average_times(average_times):
    plt.figure(figsize=(10, 5))
    plt.bar(average_times.keys(), average_times.values(), color='skyblue')
    plt.xlabel("File")
    plt.ylabel("Thời gian trung bình (giây)")
    plt.title("Thời gian trung bình xử lý từng file")
    plt.xticks(rotation=45, ha='right')
    plt.show()

# Mở hộp thoại để chọn file log
file_path = select_file()

if file_path:
    parsed_data = parse_log_file(file_path)
    
    # Tính trung bình thời gian theo từng file
    average_times = calculate_average_time(parsed_data)
    print("Thời gian trung bình của từng file:")
    for file_name, avg_time in average_times.items():
        print(f"{file_name}: {avg_time} giây")
    
    # Tính trung bình tổng thể
    overall_average = calculate_overall_average(average_times)
    print(f"\nTổng trung bình thời gian của tất cả file: {overall_average} giây")
    
    # Vẽ biểu đồ
    plot_average_times(average_times)
else:
    print("Bạn chưa chọn file nào!")
