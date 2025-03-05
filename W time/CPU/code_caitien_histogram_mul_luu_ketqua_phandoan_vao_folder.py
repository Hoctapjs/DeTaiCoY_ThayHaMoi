import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from skimage import io, color
from scipy.signal import find_peaks
import time
import logging
import os

def kiemThuChayNhieuLan(i, name, folder_path):
    # Kiểm tra xem thư mục có tồn tại không
    if not os.path.isdir(folder_path):
        print(f"❌ Thư mục {folder_path} không tồn tại!")
        return
    
    # Lấy danh sách tất cả ảnh trong thư mục
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    if not image_files:
        print(f"❌ Không tìm thấy file ảnh nào trong {folder_path}!")
        return

    for idx, file_name in enumerate(image_files, start=1):
        image_path = os.path.join(folder_path, file_name)

        # Tạo file log riêng cho từng lần chạy
        log_file = f"{name}_{i}_{idx}.txt"
        save_image_name = f"{name}_{i}_{idx}.png"

        
        # Cấu hình logging
        logging.basicConfig(filename=log_file, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

        print(f"📷 Đang xử lý ảnh {idx}: {image_path}")
        
        # Gọi hàm xử lý ảnh
        normalized_cuts(i, file_name, image_path, save_image_name)

def find_peaks_with_conditions(histogram, delta_threshold, dist_threshold):
    peaks, _ = find_peaks(histogram)
    valid_peaks = []
    
    for i in range(len(peaks)):
        is_valid = True
        for j in range(len(peaks)):
            if i != j:
                delta = abs(histogram[peaks[i]] - histogram[peaks[j]])
                dist = abs(peaks[i] - peaks[j])
                if delta < delta_threshold or dist < dist_threshold:
                    is_valid = False
                    break
        
        if is_valid:
            valid_peaks.append(peaks[i])
    
    return np.array(valid_peaks)

def determine_k_from_histogram(image):
    gray_image = color.rgb2gray(image) if len(image.shape) == 3 else image
    histogram, _ = np.histogram(gray_image, bins=256, range=(0, 1))
    delta_threshold = np.max(histogram) * 0.1
    dist_threshold = 20
    valid_peaks = find_peaks_with_conditions(histogram, delta_threshold, dist_threshold)
    return max(2, len(valid_peaks))

def determine_max_k(image):
    k = determine_k_from_histogram(image)
    h, w, _ = image.shape
    return max(2, min(k, int(np.sqrt(h * w) / 10)))

def compute_weight_matrix(image, sigma_i=0.1, sigma_x=10):
    h, w, c = image.shape
    coords = np.array(np.meshgrid(range(h), range(w))).reshape(2, -1).T
    features = image.reshape(-1, c)
    return rbf_kernel(features, gamma=1/(2 * sigma_i**2)) * rbf_kernel(coords, gamma=1/(2 * sigma_x**2))

def compute_laplacian(W):
    D = np.diag(W.sum(axis=1))
    return D - W, D

def compute_eigen(L, D, k):
    vals, vecs = eigsh(L, k=k, M=D, which='SM')
    return vals, vecs

def assign_labels(eigen_vectors, k):
    return KMeans(n_clusters=k, random_state=0).fit(eigen_vectors).labels_

def save_segmentation(image, labels, k, output_path):
    h, w, c = image.shape
    segmented_image = np.zeros_like(image, dtype=np.uint8)
    for i in range(k):
        mask = labels.reshape(h, w) == i
        cluster_pixels = image[mask]
        mean_color = (cluster_pixels.mean(axis=0) * 255).astype(np.uint8) if len(cluster_pixels) > 0 else np.array([0, 0, 0], dtype=np.uint8)
        segmented_image[mask] = mean_color
    io.imsave(output_path, segmented_image)

def normalized_cuts(lan, imagename, image_path, output_path):
    start_cpu = time.time()
    logging.info(f"file name: {imagename}")
    logging.info(f"Lan thu: {lan}")
    image = io.imread(image_path)
    image = color.gray2rgb(image) if image.ndim == 2 else image[:, :, :3] if image.shape[2] == 4 else image
    image = image / 255.0
    k = determine_max_k(image)
    start_cpu_coo = time.time()
    W = compute_weight_matrix(image)
    end_cpu_coo = time.time()

    L, D = compute_laplacian(W)
    _, vecs = compute_eigen(L, D, k)
    labels = assign_labels(vecs, k)
    save_segmentation(image, labels, k, output_path)
    end_cpu = time.time()
    logging.info(f"Thoi gian: {end_cpu - start_cpu} giay")
    logging.info(f"Thoi gian ma tran W: {end_cpu_coo - start_cpu_coo} giay")
    return labels, k
