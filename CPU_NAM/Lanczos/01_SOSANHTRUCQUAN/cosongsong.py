import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse.linalg import eigsh, LinearOperator
from scipy.sparse import diags, coo_matrix, csr_matrix
from sklearn.cluster import KMeans
from skimage import io, color
import time
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from scipy.sparse import isspmatrix
import logging
import os
from joblib import Parallel, delayed
from numba import jit, prange
import pandas as pd

# Hàm kiểm thử nhiều lần
def kiemThuChayNhieuLan(i, name, folder_path):
    if not os.path.isdir(folder_path):
        print(f"❌ Thư mục {folder_path} không tồn tại!")
        return
    
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    if not image_files:
        print(f"❌ Không tìm thấy file ảnh nào trong {folder_path}!")
        return

    excel_data = []
    
    for idx, file_name in enumerate(image_files, start=1):
        image_path = os.path.join(folder_path, file_name)
        log_file = f"{name}_{i}_{idx}.txt"
        save_image_name = f"{name}_{i}_{idx}.png"
        logging.basicConfig(filename=log_file, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        print(f"📷 Đang xử lý ảnh {idx}: {image_path}")
        
        try:
            labels, k, lanczos_time = normalized_cuts(i, file_name, image_path, save_image_name)
            excel_data.append({
                "Lần chạy": i,
                "Ảnh số": idx,
                "Tên ảnh": file_name,
                "Thời gian tổng Lanczos (s)": lanczos_time
            })
            print(f"Thời gian Lanczos cho {file_name}: {lanczos_time} giây")
        except Exception as e:
            print(f"❌ Lỗi khi xử lý ảnh {file_name}: {str(e)}")
            logging.error(f"Lỗi khi xử lý ảnh {file_name}: {str(e)}")
    
    if excel_data:
        try:
            df = pd.DataFrame(excel_data)
            excel_filename = f"{name}_lanczos_time_run_{i}.xlsx"
            print(f"Chuẩn bị ghi file Excel: {excel_filename}")
            df.to_excel(excel_filename, index=False, engine='openpyxl')
            print(f"Đã ghi thời gian Lanczos vào file: {excel_filename}")
        except Exception as e:
            print(f"❌ Lỗi khi ghi file Excel {excel_filename}: {str(e)}")
            logging.error(f"Lỗi khi ghi file Excel {excel_filename}: {str(e)}")
    else:
        print(f"❌ Không có dữ liệu để ghi vào file Excel cho lần chạy {i}")

# Tính ma trận trọng số
def compute_weight_matrix(image, sigma_i=0.1, sigma_x=10):
    h, w, c = image.shape
    coords = np.array(np.meshgrid(range(h), range(w))).reshape(2, -1).T
    features = image.reshape(-1, c)
    W_features = rbf_kernel(features, gamma=1/(2 * sigma_i**2))
    W_coords = rbf_kernel(coords, gamma=1/(2 * sigma_x**2))
    W = W_features * W_coords
    W_sparse = coo_matrix(W)
    return W_sparse

# Tính ma trận Laplace
def compute_laplacian(W_sparse):
    D_diag = W_sparse.sum(axis=1).A.flatten()
    D = np.diag(D_diag)
    L = D - W_sparse.toarray()  # Ma trận dày đặc để đồng bộ với phiên bản không song song
    return L, D

# Tích vô hướng với Numba (logic giống phiên bản không song song)
@jit(nopython=True)
def handle_dot(a, b):
    result = 0.0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result

# Nhân ma trận-vector với Numba (logic giống phiên bản không song song, nhưng song song hóa)
@jit(nopython=True)
def matrix_vector_product(A_data, A_indices, A_indptr, v, n, num_threads=os.cpu_count()):
    result = np.zeros(n)
    if n > 5000:  # Ngưỡng kích hoạt song song hóa
        chunk_size = n // num_threads
        for t in prange(num_threads):
            start = t * chunk_size
            end = (t + 1) * chunk_size if t < num_threads - 1 else n
            for i in range(start, end):
                for j in range(A_indptr[i], A_indptr[i + 1]):
                    result[i] += A_data[j] * v[A_indices[j]]
    else:
        for i in range(n):
            for j in range(A_indptr[i], A_indptr[i + 1]):
                result[i] += A_data[j] * v[A_indices[j]]
    return result

# Lanczos cho ma trận dày đặc (NumPy)
def lanczos_dense(A, v, m):
    n = len(v)
    V = np.zeros((m, n))
    T = np.zeros((m, m))
    V[0, :] = v / np.linalg.norm(v)
    w = A @ V[0, :]  # Giữ nguyên để đồng bộ giao diện với không song song
    alpha = handle_dot(w, V[0, :])  # Dùng handle_dot
    w = w - alpha * V[0, :]
    T[0, 0] = alpha
    
    for j in range(1, m):
        beta = np.linalg.norm(w)
        if beta < 1e-10:
            break
        V[j, :] = w / beta
        w = A @ V[j, :]  # Giữ nguyên để đồng bộ giao diện
        alpha = handle_dot(w, V[j, :])
        w = w - alpha * V[j, :] - beta * V[j-1, :]
        
        T[j, j] = alpha
        T[j-1, j] = beta
        T[j, j-1] = beta
    
    return T, V

# Lanczos cho ma trận thưa (Numba, logic giống phiên bản không song song)
@jit(nopython=True)
def lanczos_sparse(A_data, A_indices, A_indptr, v, m, n):
    V = np.zeros((m, n))
    T = np.zeros((m, m))
    V[0, :] = v / np.linalg.norm(v)
    w = matrix_vector_product(A_data, A_indices, A_indptr, V[0, :], n)  # Dùng matrix_vector_product
    alpha = handle_dot(w, V[0, :])                                      # Dùng handle_dot
    w = w - alpha * V[0, :]
    T[0, 0] = alpha
    
    for j in range(1, m):
        beta = np.linalg.norm(w)
        if beta < 1e-10:
            break
        V[j, :] = w / beta
        w = matrix_vector_product(A_data, A_indices, A_indptr, V[j, :], n)  # Dùng matrix_vector_product
        alpha = handle_dot(w, V[j, :])                                      # Dùng handle_dot
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
    
    n = L.shape[0]
    if n < 10000:  # Dữ liệu nhỏ: dùng ma trận dày đặc
        start_lanczos = time.time()
        T, V = lanczos_dense(L_normalized, v0, m=k+5)
        end_lanczos = time.time()
    else:  # Dữ liệu lớn: dùng CSR và song song hóa
        if not isspmatrix(L_normalized):
            L_normalized = csr_matrix(L_normalized)
        L_csr = L_normalized.tocsr()
        start_lanczos = time.time()
        T, V = lanczos_sparse(L_csr.data, L_csr.indices, L_csr.indptr, v0, m=k+5, n=n)
        end_lanczos = time.time()
    
    lanczos_time = end_lanczos - start_lanczos
    logging.info(f"Thoi gian Lanczos: {lanczos_time} giay, n={n}")
    
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
    start_cpu_coo = time.time()
    W = compute_weight_matrix(image)
    end_cpu_coo = time.time()
    L, D = compute_laplacian(W)
    vecs, lanczos_time = compute_eigen(L, D, k)
    labels = assign_labels(vecs, k)
    save_segmentation(image, labels, k, output_path)
    end_cpu = time.time()
    logging.info(f"Thoi gian: {end_cpu - start_cpu} giay")
    logging.info(f"Thoi gian ma tran W: {end_cpu_coo - start_cpu_coo} giay")
    return labels, k, lanczos_time

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