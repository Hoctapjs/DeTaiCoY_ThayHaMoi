import numpy as np
from sklearn.cluster import KMeans
from skimage import io, color
import time
import os
import pandas as pd
from datetime import datetime
import logging
from scipy.sparse import diags
from joblib import Parallel, delayed
import numpy as np
import time

import cupy as cp  # Thay thế NumPy bằng CuPy


# CUDA Kernel để tính RBF Kernel song song
rbf_kernel_cuda = cp.RawKernel(r'''
extern "C" __global__
void rbf_kernel(const double* X, double* W, int n, int d, double gamma) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double dist = 0.0;
                for (int k = 0; k < d; k++) {
                    double diff = X[i * d + k] - X[j * d + k];
                    dist += diff * diff;
                }
                W[i * n + j] = exp(-gamma * dist);
            }
        }
    }
}
''', 'rbf_kernel')

def compute_rbf_matrix(X, gamma):
    n, d = X.shape
    X_gpu = cp.asarray(X, dtype=cp.float64)
    W_gpu = cp.zeros((n, n), dtype=cp.float64)

    # Chạy với chỉ 1 thread (1 block, 1 thread)
    rbf_kernel_cuda((1,), (1,), (X_gpu, W_gpu, n, d, gamma))

    return W_gpu



# 1. Tinh ma tran trong so
def compute_weight_matrix(image, sigma_i=0.1, sigma_x=10):
    h, w, c = image.shape
    coords = cp.array(cp.meshgrid(cp.arange(h), cp.arange(w))).reshape(2, -1).T  # Tọa độ (x, y)
    features = cp.array(image.reshape(-1, c))  # Chuyển toàn bộ đặc trưng lên GPU

    gamma_i = 1 / (2 * sigma_i**2)
    gamma_x = 1 / (2 * sigma_x**2)

    start_features = time.time()
    W_features = compute_rbf_matrix(features, gamma_i)
    cp.cuda.Stream.null.synchronize()
    end_features = time.time()

    W_features_time = end_features - start_features

    start_coords = time.time()
    W_coords = compute_rbf_matrix(coords, gamma_x)
    cp.cuda.Stream.null.synchronize()
    end_coords = time.time()

    W_coords_time = end_coords - start_coords

    W = cp.multiply(W_features, W_coords)
    return W, W_features_time, W_coords_time

# Tính ma trận Laplacian
def compute_laplacian(W):
    D = cp.diag(W.sum(axis=1))
    L = D - W
    return L, D

# Hàm tích vô hướng
def handle_dot(a, b):
    return cp.dot(a, b)

# Hàm nhân ma trận-vector
def matrix_vector_product(A, v):
    return A @ v  # Sử dụng phép nhân ma trận-vector của CuPy

# Thuật toán Lanczos với trực giao hóa chặt chẽ
def Lanczos(A, v, m):
    n = len(v)
    V = cp.zeros((m, n), dtype=cp.float64)
    T = cp.zeros((m, m), dtype=cp.float64)
    V[0, :] = v / cp.linalg.norm(v)
    
    w = matrix_vector_product(A, V[0, :])
    alpha = handle_dot(w, V[0, :])
    w = w - alpha * V[0, :]
    T[0, 0] = alpha
    
    for j in range(1, m):
        beta = cp.linalg.norm(w)
        if beta < 1e-10:
            break
        V[j, :] = w / beta
        # Trực giao hóa (Gram-Schmidt) chặt chẽ
        for i in range(j):
            proj = handle_dot(V[j, :], V[i, :])
            V[j, :] -= proj * V[i, :]
        norm_vj = cp.linalg.norm(V[j, :])
        if norm_vj < 1e-10:
            break
        V[j, :] /= norm_vj
        
        w = matrix_vector_product(A, V[j, :])
        alpha = handle_dot(w, V[j, :])
        w = w - alpha * V[j, :] - (beta * V[j-1, :] if j > 0 else 0)
        
        T[j, j] = alpha
        if j > 0:
            T[j-1, j] = beta
            T[j, j-1] = beta
    
    # Đảm bảo T đối xứng
    T = (T + T.T) / 2
    return T[:j+1, :j+1], V[:j+1, :]

import cupy as cp
import cupyx.scipy.sparse as cpsp
import numpy as np
import scipy.sparse as sp
import cupyx.scipy.sparse as cpx_sparse

def compute_eigen(L, D, k=2):
    D_diag = D.diagonal().copy()
    D_diag[D_diag < 1e-10] = 1e-10
    
    D_diag_np = cp.asnumpy(D_diag)
    D_inv_sqrt_cp = cpx_sparse.diags(1.0 / cp.sqrt(D_diag))
    L_normalized = D_inv_sqrt_cp @ L @ D_inv_sqrt_cp
    
    v0 = cp.random.rand(L.shape[0])
    v0 /= cp.linalg.norm(v0)
    
    m = min(k + 100, L.shape[0])
    T, V = Lanczos(L_normalized, v0, m)
    
    eigvals, eigvecs_T = cp.linalg.eigh(T)
    idx = cp.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs_T = eigvecs_T[:, idx]
    
    eigvecs_T = eigvecs_T[:, 1:k+1]
    
    print("D_inv_sqrt shape:", D_inv_sqrt_cp.shape)
    print("V shape:", V.shape)
    print("eigvecs_T shape:", eigvecs_T.shape)
    print(f"V.T shape: {V.T.shape}, eigvecs_T shape: {eigvecs_T.shape}")

    eigvecs_original = cp.matmul(V.T, eigvecs_T)  # Dùng cp.matmul thay vì @

    if D_inv_sqrt_cp.shape[0] == eigvecs_original.shape[0]:
        eigvecs_original = D_inv_sqrt_cp @ eigvecs_original
    
    for i in range(eigvecs_original.shape[1]):
        norm = cp.linalg.norm(eigvecs_original[:, i])
        if norm > 1e-10:
            eigvecs_original[:, i] /= norm

    lanczos_time = 0
    
    return eigvecs_original, lanczos_time


# Gán nhãn
def assign_labels(eigen_vectors, k):
    # Chuyển từ CuPy về NumPy nếu cần
    if isinstance(eigen_vectors, cp.ndarray):
        eigen_vectors = eigen_vectors.get()

    return KMeans(n_clusters=k, random_state=42, n_init=30).fit(eigen_vectors).labels_


# Lưu file .seg
def save_seg_file(labels, image_shape, output_path, image_name="image"):
    h, w = image_shape[:2]
    labels_2d = labels.reshape(h, w)
    unique_labels = np.unique(labels)
    segments = len(unique_labels)

    header = [
        "format ascii cr",
        f"date {datetime.now().strftime('%a %b %d %H:%M:%S %Y')}",
        f"image {image_name}",
        "user 1102",
        f"width {w}",
        f"height {h}",
        f"segments {segments}",
        "gray 0",
        "invert 0",
        "flipflop 0",
        "data"
    ]

    data_lines = []
    for row in range(h):
        row_labels = labels_2d[row, :]
        start_col = 0
        current_label = row_labels[0]

        for col in range(1, w):
            if row_labels[col] != current_label:
                data_lines.append(f"{int(current_label)} {row} {start_col} {col - 1}")
                start_col = col
                current_label = row_labels[col]
        data_lines.append(f"{int(current_label)} {row} {start_col} {w - 1}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n")
        f.write("\n".join(data_lines) + "\n")
    print(f"✅ File SEG đã lưu: {output_path}")

# Hàm chính xử lý ảnh
def normalized_cuts(lan, imagename, image_path, output_path):
    start_cpu = time.time()
    image = io.imread(image_path)
    image = color.gray2rgb(image) if image.ndim == 2 else image[:, :, :3] if image.shape[2] == 4 else image
    image = image / 255.0
    k = 2  # Sử dụng k=2 để khớp với kết quả quả táo

    W, W_f, W_c = compute_weight_matrix(image, sigma_i=0.5, sigma_x=15.0)
    W_all = W_f + W_c
    L, D = compute_laplacian(W)
    vecs, lanczos_time = compute_eigen(L, D, k)
    labels = assign_labels(vecs, k)

    seg_output_path = f"{imagename}_segmentation_{lan}.seg"
    save_seg_file(labels, image.shape, seg_output_path, imagename)

    end_cpu = time.time()
    total_cpu_time = end_cpu - start_cpu
    return total_cpu_time, W_f, W_c, W_all

# Hàm chạy nhiều ảnh và lưu kết quả
def kiemThuChayNhieuLan(i, name, folder_path, output_excel_base="results"):
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

        imagename = os.path.splitext(file_name)[0]
        total_time, wf_time, wc_time, W_all = normalized_cuts(i, imagename, image_path, output_excel_base)
        results.append([i, idx, file_name, wf_time, wc_time, W_all])

    df = pd.DataFrame(results, columns=["Lần chạy", "Ảnh số", "Tên ảnh", "Thời gian W đặc trưng (s)", "Thời gian W tọa độ (s)", "Thời gian W All"])
    output_excel = f"{output_excel_base}_{name}_{i}.xlsx"
    df.to_excel(output_excel, index=False, engine='openpyxl')
    print(f"✅ Kết quả Excel đã lưu vào {output_excel}")