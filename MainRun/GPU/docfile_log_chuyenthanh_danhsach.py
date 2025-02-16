import re
import tkinter as tk
from tkinter import filedialog
from collections import defaultdict

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
                current_entry["thoi_gian_w"] = round(float(match_time_coo.group(1)), 2)

        if current_entry:
            entries.append(current_entry)

    return entries

def calculate_average_time(entries):
    time_data = defaultdict(lambda: {"total_time": 0, "total_time_coo": 0, "count": 0})

    for entry in entries:
        file_name = entry["file_name"]
        time_data[file_name]["total_time"] += entry["thoi_gian"]
        time_data[file_name]["total_time_coo"] += entry["thoi_gian_w"]
        time_data[file_name]["count"] += 1

    averages = {
        file: {
            "average_thoi_gian": round(data["total_time"] / data["count"], 2),
            "average_thoi_gian_w": round(data["total_time_coo"] / data["count"], 2),
        }
        for file, data in time_data.items()
    }

    return averages

def calculate_overall_average(average_times):
    total_avg_thoi_gian = 0
    total_avg_thoi_gian_w = 0
    count = len(average_times)

    for avg_data in average_times.values():
        total_avg_thoi_gian += avg_data["average_thoi_gian"]
        total_avg_thoi_gian_w += avg_data["average_thoi_gian_w"]

    overall_avg = {
        "overall_average_thoi_gian": round(total_avg_thoi_gian / count, 2),
        "overall_average_thoi_gian_w": round(total_avg_thoi_gian_w / count, 2),
    }

    return overall_avg

# Mở hộp thoại để chọn file log
file_path = select_file()

if file_path:
    parsed_data = parse_log_file(file_path)

    # Hiển thị kết quả
    for entry in parsed_data:
        print(entry)

    # Tính trung bình thời gian theo từng file
    average_times = calculate_average_time(parsed_data)

    print("Hiển thị thời gian trung bình của từng file")

    # Hiển thị kết quả trung bình theo file
    for file_name, avg_data in average_times.items():
        print(f"{file_name}: Trung bình thời gian = {avg_data['average_thoi_gian']:.2f} giây, "
              f"Trung bình thời gian W = {avg_data['average_thoi_gian_w']:.2f} giây")

    # Tính trung bình tổng thể từ danh sách trung bình đã có (trung bình của tất cả file)
    overall_average = calculate_overall_average(average_times)

    print("Hiển thị thời gian trung bình của folder")

    # Hiển thị kết quả trung bình tổng thể
    print(f"\nTổng trung bình thời gian cả folder: {overall_average['overall_average_thoi_gian']} giây")
    print(f"Tổng trung bình thời gian W cả folder: {overall_average['overall_average_thoi_gian_w']} giây")
else:
    print("Bạn chưa chọn file nào!")


""" chèn luôn file vẽ vào file này để tận dụng các danh sách dữ liệu đã có để vẽ """
