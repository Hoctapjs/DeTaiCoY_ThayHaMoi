import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from skimage import io, color
import time
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import logging
from scipy.sparse import coo_matrix #chuyển sang ma trận coo
from joblib import Parallel, delayed
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

    

# 1. Tinh ma tran trong so
def compute_weight_matrix(image, sigma_i=0.1, sigma_x=10):
    h, w, c = image.shape
    coords = np.array(np.meshgrid(range(h), range(w))).reshape(2, -1).T  # Toa do (x, y)
    features = image.reshape(-1, c)  # Dac trung mau
    
    logging.info(f"Kich thuoc anh: {h}x{w}x{c}")
    logging.info(f"Kich thuoc dac trung mau: {features.shape}, Kich thuoc toa do: {coords.shape}")
    logging.info(f"Dac trung mau:\n{features[:9, :9]}")
    logging.info(f"Toa do:\n{coords[:9, :9]}")

    
    # Tinh do tuong dong ve dac trung va khong gian
    W_features = rbf_kernel(features, gamma=1/(2 * sigma_i**2))
    W_coords = rbf_kernel(coords, gamma=1/(2 * sigma_x**2))
    W = W_features * W_coords
    
    logging.info(f"Kich thuoc ma tran trong so (W): {W.shape}")
    logging.info(f"Kich thuoc ma tran dac trung mau: {W_features.shape}, Kich thuoc ma tran toa do: {W_coords.shape}")
    logging.info(f"Mau cua W_features (9x9 phan tu dau):\n{W_features[:9, :9]}")
    logging.info(f"Mau cua W_coords (9x9 phan tu dau):\n{W_coords[:9, :9]}")
    logging.info(f"Mau cua W (9x9 phan tu dau):\n{W[:9, :9]}")

    # Chuyen ma tran W sang dang ma tran thua COO
    start_coo = time.time() 
    W_sparse = coo_matrix(W)
    end_coo = time.time()  

    logging.info(f"Thời gian xử lý (không song song): {end_coo - start_coo:.4f} giây")
    logging.info(f"Kich thuoc ma tran thua (COO): {W_sparse.shape}")
    logging.info(f"Mau cua ma tran thua (COO) [du lieu, hang, cot]:\n{W_sparse.data[:9]}, {W_sparse.row[:9]}, {W_sparse.col[:9]}")
    
    return W_sparse

# 1. Tinh ma tran trong so
# def compute_weight_matrix(image, sigma_i=0.1, sigma_x=10, n_jobs=-1):
#     h, w, c = image.shape
#     coords = np.array(np.meshgrid(range(h), range(w))).reshape(2, -1).T  # Tọa độ (x, y)
#     features = image.reshape(-1, c)  # Đặc trưng màu
    
#     logging.info(f"Kích thước ảnh: {h}x{w}x{c}")
#     logging.info(f"Kích thước đặc trưng màu: {features.shape}, Kích thước tọa độ: {coords.shape}")

#     # Song song hóa tính toán ma trận đặc trưng màu (W_features)
#     def compute_rbf_chunk(start, end):
#         return rbf_kernel(features[start:end], features, gamma=1/(2 * sigma_i**2))

#     n_samples = features.shape[0]
#     n_chunks = 8  # Chia thành 8 phần để chạy song song
#     chunk_size = (n_samples + n_chunks - 1) // n_chunks  # Kích thước mỗi phần
    
#     results = Parallel(n_jobs=n_jobs)(
#         delayed(compute_rbf_chunk)(i, min(i + chunk_size, n_samples))
#         for i in range(0, n_samples, chunk_size)
#     )
    
#     W_features = np.vstack(results)  # Ghép lại thành ma trận hoàn chỉnh
    
#     # Tính ma trận tọa độ W_coords
#     W_coords = rbf_kernel(coords, gamma=1/(2 * sigma_x**2))

#     # Nhân hai ma trận để lấy ma trận trọng số cuối cùng
#     W = W_features * W_coords

#     logging.info(f"Kích thước ma trận trọng số (W): {W.shape}")

#     # Chuyển sang dạng ma trận thưa COO
#     start_coo = time.time()  
#     W_sparse = coo_matrix(W)
#     end_coo = time.time() 

#     logging.info(f"Thời gian xử lý (song song hóa): {end_coo - start_coo:.4f} giây")
    
#     return W_sparse


# 2. Tinh ma tran Laplace
def compute_laplacian(W_sparse):
    # Tạo ma trận đường chéo từ tổng các hàng
    D_diag = W_sparse.sum(axis=1).A.flatten() 
    D = np.diag(D_diag)  # Ma trận đường chéo
    L = D - W_sparse # L = D - W
    logging.info("Kich thuoc ma tran duong cheo (D): %s", D.shape)
    logging.info("Mau cua D (9 phan tu dau):\n%s", D_diag[:9])  # In phần tử trên đường chéo
    logging.info("Kich thuoc ma tran Laplace (L): %s", L.shape)
    logging.info("Mau cua L (9x9 phan tu dau):\n%s", L[:9, :9])
    return L, D


# 3. Giai bai toan tri rieng
def compute_eigen(L, D, k=2):
    # Tìm các trị riêng nhỏ nhất (Smallest Magnitude)
    eigvals, eigvecs = eigsh(L, k=k, which='SA')  
    return eigvecs


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
    k = 2
    start_cpu_coo = time.time()
    W = compute_weight_matrix(image)
    end_cpu_coo = time.time()

    L, D = compute_laplacian(W)
    vecs = compute_eigen(L, D, k)
    labels = assign_labels(vecs, k)
    save_segmentation(image, labels, k, output_path)
    end_cpu = time.time()
    logging.info(f"Thoi gian: {end_cpu - start_cpu} giay")
    logging.info(f"Thoi gian ma tran W: {end_cpu_coo - start_cpu_coo} giay")
    return labels, k
