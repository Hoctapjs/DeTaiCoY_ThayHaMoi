import re
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import filedialog

def extract_data(filename):
    data = {}
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        current_image = None
        for line in lines:
            match_file = re.search(r'file name: (.+)', line)
            match_time = re.search(r'Thoi gian: ([\d\.]+) giay', line)
            match_time_w = re.search(r'Thoi gian W: ([\d\.]+) giay', line)
            
            if match_file:
                current_image = match_file.group(1)
                data[current_image] = {'Thoi gian': [], 'Thoi gian W': []}
            elif match_time and current_image:
                data[current_image]['Thoi gian'].append(float(match_time.group(1)))
            elif match_time_w and current_image:
                data[current_image]['Thoi gian W'].append(float(match_time_w.group(1)))
    return data

def compute_averages(data):
    avg_data = {}
    total_time = []
    total_time_w = []
    for key, values in data.items():
        avg_time = np.mean(values['Thoi gian'])
        avg_time_w = np.mean(values['Thoi gian W'])
        avg_data[key] = {'Thoi gian': avg_time, 'Thoi gian W': avg_time_w}
        total_time.extend(values['Thoi gian'])
        total_time_w.extend(values['Thoi gian W'])
    
    overall_avg_time = np.mean(total_time)
    overall_avg_time_w = np.mean(total_time_w)
    
    return avg_data, overall_avg_time, overall_avg_time_w

def plot_comparison(avg_data1, avg_data2, label1, label2, overall_avg1, overall_avg_w1, overall_avg2, overall_avg_w2):
    labels = sorted(avg_data1.keys())
    times1 = [avg_data1[label]['Thoi gian'] for label in labels]
    times2 = [avg_data2[label]['Thoi gian'] for label in labels]
    times_w1 = [avg_data1[label]['Thoi gian W'] for label in labels]
    times_w2 = [avg_data2[label]['Thoi gian W'] for label in labels]
    
    x = np.arange(len(labels))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width*1.5, times1, width, label=f'{label1} - Thoi gian')
    ax.bar(x - width/2, times_w1, width, label=f'{label1} - Thoi gian W')
    ax.bar(x + width/2, times2, width, label=f'{label2} - Thoi gian')
    ax.bar(x + width*1.5, times_w2, width, label=f'{label2} - Thoi gian W')
    
    ax.axhline(y=overall_avg1, color='r', linestyle='--', label=f'{label1} Overall Avg: {overall_avg1:.2f}s')
    ax.axhline(y=overall_avg_w1, color='g', linestyle='--', label=f'{label1} Overall W Avg: {overall_avg_w1:.2f}s')
    ax.axhline(y=overall_avg2, color='b', linestyle='--', label=f'{label2} Overall Avg: {overall_avg2:.2f}s')
    ax.axhline(y=overall_avg_w2, color='y', linestyle='--', label=f'{label2} Overall W Avg: {overall_avg_w2:.2f}s')
    
    ax.set_xlabel('Image Size')
    ax.set_ylabel('Time (s)')
    ax.set_title('Comparison of Processing Times')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    
    plt.show()

# Chọn file bằng hộp thoại hai lần
def select_file(prompt):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=prompt, filetypes=[("Text Files", "*.txt")])
    return file_path

file1 = select_file("Select the first log file")
file2 = select_file("Select the second log file")

if not file1 or not file2:
    print("Please select two log files.")
else:
    data1 = extract_data(file1)
    data2 = extract_data(file2)
    avg_data1, overall_avg1, overall_avg_w1 = compute_averages(data1)
    avg_data2, overall_avg2, overall_avg_w2 = compute_averages(data2)
    plot_comparison(avg_data1, avg_data2, "File 1", "File 2", overall_avg1, overall_avg_w1, overall_avg2, overall_avg_w2)