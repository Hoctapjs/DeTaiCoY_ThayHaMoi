import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse.linalg import eigsh, LinearOperator
from scipy.sparse import diags, coo_matrix
from sklearn.cluster import KMeans
from skimage import io, color
import time
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from scipy.sparse import isspmatrix
import logging
import os
import pandas as pd

# Hàm kiểm thử nhiều lần
def kiemThuChayNhieuLan(i, name, folder_path, output_excel="results.xlsx"):
    if not os.path.isdir(folder_path):
        print(f"❌ Thư mục {folder_path} không tồn tại!")
        return
    
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    if not image_files:
        print(f"❌ Không tìm thấy file ảnh nào trong {folder_path}!")
        return

    results = []
    for idx, file_name in enumerate(image_files, start=1):
        image_path = os.path.join(folder_path, file_name)
        print(f"📷 Đang xử lý ảnh {idx}: {image_path}")
        
        _, lanczos_time = normalized_cuts(i, file_name, image_path, output_excel)
        results.append([i, idx, file_name, lanczos_time])

    df = pd.DataFrame(results, columns=["Lần chạy", "Ảnh số", "Tên ảnh", "Thời gian tổng Lanczos (s)"])
    output_excel = f"result_{name}_{i}.xlsx"
    df.to_excel(output_excel, index=False, engine='openpyxl')
    print(f"✅ Kết quả đã lưu vào {output_excel}")

# Tính ma trận trọng số
def compute_weight_matrix(image, sigma_i=0.1, sigma_x=10):
    h, w, c = image.shape
    coords = np.array(np.meshgrid(range(h), range(w))).reshape(2, -1).T  # Toa do (x, y)
    features = image.reshape(-1, c)  # Dac trung mau
    
    # Tinh do tuong dong ve dac trung va khong gian
    W_features = rbf_kernel(features, gamma=1/(2 * sigma_i**2))
    W_coords = rbf_kernel(coords, gamma=1/(2 * sigma_x**2))
    W = W_features * W_coords
    
    return W

# Tính ma trận Laplace
def compute_laplacian(W_sparse):
    # Kiểm tra xem W_sparse có phải là ma trận thưa không
    if isspmatrix(W_sparse):
        D_diag = W_sparse.sum(axis=1).A.flatten()  # Nếu là ma trận thưa, dùng .A
    else:
        D_diag = W_sparse.sum(axis=1).flatten()    # Nếu là mảng dày đặc, không cần .A
    
    D = np.diag(D_diag)
    L = D - W_sparse  # Nếu W_sparse là ndarray, không cần .toarray()
    return L, D

# Hàm tích vô hướng (logic giống phiên bản song song, nhưng không dùng Numba)
def handle_dot(a, b):
    result = 0.0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result

# Hàm nhân ma trận-vector (logic giống phiên bản song song, nhưng dùng ma trận dày đặc)
def matrix_vector_product(A, v):
    n = A.shape[0]
    result = np.zeros(n)
    for i in range(n):
        for j in range(A.shape[1]):  # Duyệt qua các cột của A
            result[i] += A[i, j] * v[j]
    return result

# Thuật toán Lanczos (giữ logic giống phiên bản song song)
def Lanczos(A, v, m):
    n = len(v)
    V = np.zeros((m, n))
    T = np.zeros((m, m))
    V[0, :] = v / np.linalg.norm(v)
    
    w = matrix_vector_product(A, V[0, :])  # Nhân ma trận-vector
    alpha = handle_dot(w, V[0, :])        # Tích vô hướng
    
    w = w - alpha * V[0, :]
    T[0, 0] = alpha
    
    for j in range(1, m):
        beta = np.linalg.norm(w)
        if beta < 1e-10:
            break
        V[j, :] = w / beta
        w = matrix_vector_product(A, V[j, :])  # Nhân ma trận-vector
        alpha = handle_dot(w, V[j, :])         # Tích vô hướng
        w = w - alpha * V[j, :] - beta * V[j-1, :]
        
        T[j, j] = alpha
        T[j-1, j] = beta
        T[j, j-1] = beta
    
    return T, V

# Tính trị riêng và vector riêng
def compute_eigen(L, D, k=2):
    D_diag = D.diagonal().copy()
    D_diag[D_diag < 1e-10] = 1e-10
    D_inv_sqrt = diags(1.0 / np.sqrt(D_diag))
    L_normalized = D_inv_sqrt @ L @ D_inv_sqrt
    
    v0 = np.random.rand(L.shape[0])
    v0 /= np.linalg.norm(v0)

    lanczos_time_start = time.time()
    T, V = Lanczos(L_normalized, v0, m=k+5)
    lanczos_time_end = time.time()

    lanczos_time = lanczos_time_end - lanczos_time_start
    logging.info(f"Thoi gian lanczos khong song song: {lanczos_time:.6f} giay")
    
    eigvals, eigvecs_T = np.linalg.eig(T[:k, :k])
    eigvecs_original = D_inv_sqrt @ (V[:k, :].T @ eigvecs_T)
    
    return eigvecs_original, lanczos_time

# Gán nhãn
def assign_labels(eigen_vectors, k):
    return KMeans(n_clusters=k, random_state=0).fit(eigen_vectors).labels_

# Lưu kết quả phân đoạn
def save_segmentation(image, labels, k, output_path):
    h, w, c = image.shape
    segmented_image = np.zeros_like(image, dtype=np.uint8)
    for i in range(k):
        mask = labels.reshape(h, w) == i
        cluster_pixels = image[mask]
        mean_color = (cluster_pixels.mean(axis=0) * 255).astype(np.uint8) if len(cluster_pixels) > 0 else np.array([0, 0, 0], dtype=np.uint8)
        segmented_image[mask] = mean_color
    io.imsave(output_path, segmented_image)

# Hàm chính: Normalized Cuts
def normalized_cuts(lan, imagename, image_path, output_path):
    start_cpu = time.time()
    logging.info(f"file name: {imagename}")
    logging.info(f"Lan thu: {lan}")

    image = io.imread(image_path)
    image = color.gray2rgb(image) if image.ndim == 2 else image[:, :, :3] if image.shape[2] == 4 else image
    image = image / 255.0
    k = 2

    W = compute_weight_matrix(image)
    L, D = compute_laplacian(W)
    vecs, lanczos_time = compute_eigen(L, D, k)
    labels = assign_labels(vecs, k)
    save_segmentation(image, labels, k, output_path)

    end_cpu = time.time()
    total_cpu_time = end_cpu - start_cpu

    logging.info(f"⏳ Tổng thời gian: {total_cpu_time:.6f} giây")
    logging.info(f"⏳ Thời gian Lanczos: {lanczos_time:.6f} giây")

    return total_cpu_time, lanczos_time

# Mở hộp thoại chọn file
def open_file_dialog():
    root = Tk()
    root.withdraw()
    file_path = askopenfilename(title="Chon anh", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
    return file_path

# Chạy thử nghiệm
if __name__ == "__main__":
    image_path = open_file_dialog()
    if image_path:
        logging.info(f"Da chon anh: {image_path}")
        normalized_cuts(1, "test", image_path, "output.png")
    else:
        logging.info("Khong co anh nao duoc chon.")