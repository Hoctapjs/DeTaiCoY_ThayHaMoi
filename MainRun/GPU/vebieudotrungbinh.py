import re
import tkinter as tk
from tkinter import filedialog
from collections import defaultdict
import matplotlib.pyplot as plt

def select_file():
    """Hiển thị hộp thoại để chọn file"""
    root = tk.Tk()
    root.withdraw()  # Ẩn cửa sổ chính
    file_path = filedialog.askopenfilename(title="Chọn file log", filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
    return file_path

def parse_log_file(file_path):
    entries = []
    current_entry = {}

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            match_file = re.search(r"file name: (\S+)", line)
            match_lan = re.search(r"Lan thu: (\d+)", line)
            match_time = re.search(r"Thoi gian: ([\d.]+)", line)
            match_time_coo = re.search(r"Thoi gian W: ([\d.]+)", line)

            if match_file:
                if current_entry:
                    entries.append(current_entry)
                current_entry = {"file_name": match_file.group(1)}
            elif match_lan:
                current_entry["lan_thu"] = int(match_lan.group(1))
            elif match_time:
                current_entry["thoi_gian"] = round(float(match_time.group(1)), 2)
            elif match_time_coo:
                current_entry["thoi_gian_coo"] = round(float(match_time_coo.group(1)), 2)

        if current_entry:
            entries.append(current_entry)

    return entries

def calculate_average_time(entries):
    time_data = defaultdict(lambda: {"total_time": 0, "total_time_coo": 0, "count": 0})

    for entry in entries:
        file_name = entry["file_name"]
        time_data[file_name]["total_time"] += entry["thoi_gian"]
        time_data[file_name]["total_time_coo"] += entry["thoi_gian_coo"]
        time_data[file_name]["count"] += 1

    averages = {
        file: {
            "average_thoi_gian": round(data["total_time"] / data["count"], 2),
            "average_thoi_gian_coo": round(data["total_time_coo"] / data["count"], 2),
        }
        for file, data in time_data.items()
    }

    return averages

def calculate_overall_average(average_times):
    total_avg_thoi_gian = 0
    total_avg_thoi_gian_coo = 0
    count = len(average_times)

    for avg_data in average_times.values():
        total_avg_thoi_gian += avg_data["average_thoi_gian"]
        total_avg_thoi_gian_coo += avg_data["average_thoi_gian_coo"]

    overall_avg = {
        "overall_average_thoi_gian": round(total_avg_thoi_gian / count, 2),
        "overall_average_thoi_gian_coo": round(total_avg_thoi_gian_coo / count, 2),
    }

    return overall_avg

def plot_bar_chart(average_times, overall_average):
    file_names = list(average_times.keys())
    avg_thoi_gian = [data["average_thoi_gian"] for data in average_times.values()]
    avg_thoi_gian_coo = [data["average_thoi_gian_coo"] for data in average_times.values()]

    x = range(len(file_names))

    plt.figure(figsize=(10, 6))
    plt.bar(x, avg_thoi_gian, width=0.4, label="Thời gian trung bình", color='b', align='center')
    plt.bar(x, avg_thoi_gian_coo, width=0.4, label="Thời gian trung bình W", color='g', align='edge')
    
    plt.axhline(y=overall_average["overall_average_thoi_gian"], color='r', linestyle='--', label="TB thời gian toàn bộ")
    plt.axhline(y=overall_average["overall_average_thoi_gian_coo"], color='y', linestyle='--', label="TB thời gian W toàn bộ")
    
    plt.xticks(x, file_names, rotation=45, ha='right')
    plt.xlabel("File")
    plt.ylabel("Thời gian (giây)")
    plt.title("Biểu đồ thời gian trung bình")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Mở hộp thoại để chọn file log
file_path = select_file()

if file_path:
    parsed_data = parse_log_file(file_path)
    average_times = calculate_average_time(parsed_data)
    overall_average = calculate_overall_average(average_times)
    plot_bar_chart(average_times, overall_average)
else:
    print("Bạn chưa chọn file nào!")
