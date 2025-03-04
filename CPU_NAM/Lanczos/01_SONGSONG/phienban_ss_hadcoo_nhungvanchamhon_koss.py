import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse.linalg import eigsh, LinearOperator
from scipy.sparse import diags
from sklearn.cluster import KMeans
from skimage import io, color
import time
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from scipy.sparse import coo_matrix
from scipy.sparse import isspmatrix, diags
import logging
import os
from joblib import Parallel, delayed  
import numba
from numba import jit, prange
from scipy.sparse import csr_matrix
import pandas as pd

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

def compute_weight_matrix(image, sigma_i=0.1, sigma_x=10):
    h, w, c = image.shape
    coords = np.array(np.meshgrid(range(h), range(w))).reshape(2, -1).T
    features = image.reshape(-1, c)
    W_features = rbf_kernel(features, gamma=1/(2 * sigma_i**2))
    W_coords = rbf_kernel(coords, gamma=1/(2 * sigma_x**2))
    W = W_features * W_coords
    W_sparse = coo_matrix(W)
    return W_sparse

def compute_laplacian(W_sparse):
    D_diag = W_sparse.sum(axis=1).A.flatten() if hasattr(W_sparse, 'toarray') else W_sparse.sum(axis=1)
    D = np.diag(D_diag)
    L = D - W_sparse.toarray() if hasattr(W_sparse, 'toarray') else D - W_sparse
    return L, D

@jit(nopython=True)
def handle_dot(a, b):
    result = 0.0
    for i in range(len(a)):  # Tắt prange vì dữ liệu nhỏ không cần song song
        result += a[i] * b[i]
    return result

@jit(nopython=True)
def matrix_vector_product(A_data, A_indices, A_indptr, v, n, num_threads=os.cpu_count()):
    result = np.zeros(n)
    if n > 5000:  # Chỉ song song với dữ liệu lớn
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

@jit(nopython=True)
def Lanczos(A_data, A_indices, A_indptr, v, m, n):
    V = np.zeros((m, n))
    T = np.zeros((m, m))
    V[0, :] = v / np.linalg.norm(v)
    w = matrix_vector_product(A_data, A_indices, A_indptr, V[0, :], n)
    alpha = handle_dot(w, V[0, :])
    w = w - alpha * V[0, :]
    T[0, 0] = alpha
    
    for j in range(1, m):
        beta = np.linalg.norm(w)
        if beta < 1e-10:
            break
        if n > 50000:  # Chỉ song song với dữ liệu lớn
            for i in prange(n):
                V[j, i] = w[i] / beta
            w = matrix_vector_product(A_data, A_indices, A_indptr, V[j, :], n)
            alpha = handle_dot(w, V[j, :])
            for i in prange(n):
                w[i] = w[i] - alpha * V[j, i] - (beta * V[j-1, i] if j > 1 else 0.0)
        else:
            for i in range(n):
                V[j, i] = w[i] / beta
            w = matrix_vector_product(A_data, A_indices, A_indptr, V[j, :], n)
            alpha = handle_dot(w, V[j, :])
            for i in range(n):
                w[i] = w[i] - alpha * V[j, i] - (beta * V[j-1, i] if j > 1 else 0.0)
        T[j, j] = alpha
        T[j-1, j] = beta
        T[j, j-1] = beta
    
    return T, V

def compute_eigen(L, D, k=2):
    D_diag = D.diagonal().copy()
    D_diag[D_diag < 1e-10] = 1e-10
    D_inv_sqrt = diags(1.0 / np.sqrt(D_diag))
    L_normalized = D_inv_sqrt @ L @ D_inv_sqrt
    
    v0 = np.random.rand(L.shape[0])
    v0 /= np.linalg.norm(v0)
    
    if not isspmatrix(L_normalized):
        L_normalized = csr_matrix(L_normalized)
    L_csr = L_normalized.tocsr()
    
    start_lanczos = time.time()
    T, V = Lanczos(L_csr.data, L_csr.indices, L_csr.indptr, v0, m=k+5, n=L.shape[0])
    end_lanczos = time.time()
    lanczos_time = end_lanczos - start_lanczos
    logging.info(f"Thoi gian Lanczos: {lanczos_time} giay, n={L.shape[0]}")  # Thêm debug n
    
    eigvals, eigvecs_T = np.linalg.eig(T[:k, :k])
    eigvecs_original = D_inv_sqrt @ (V[:k, :].T @ eigvecs_T)
    return eigvecs_original, lanczos_time

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
    vecs, lanczos_time = compute_eigen(L, D, k)
    labels = assign_labels(vecs, k)
    save_segmentation(image, labels, k, output_path)
    end_cpu = time.time()
    logging.info(f"Thoi gian: {end_cpu - start_cpu} giay")
    logging.info(f"Thoi gian ma tran W: {end_cpu_coo - start_cpu_coo} giay")
    return labels, k, lanczos_time

def open_file_dialog():
    root = Tk()
    root.withdraw()
    file_path = askopenfilename(title="Chon anh", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
    return file_path

if __name__ == "__main__":
    image_path = open_file_dialog()
    if image_path:
        logging.info(f"Da chon anh: {image_path}")
        normalized_cuts(image_path, k=3)
    else:
        logging.info("Khong co anh nao duoc chon.")