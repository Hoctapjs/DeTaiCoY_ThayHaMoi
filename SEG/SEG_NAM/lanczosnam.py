import numpy as np
from sklearn.cluster import KMeans
from skimage import io, color
import time
import os
import pandas as pd
from datetime import datetime
import logging
from scipy.sparse import diags, csr_matrix
from joblib import Parallel, delayed
from numba import jit, prange

# Hàm tính ma trận RBF song song với đầu ra là ma trận thưa
def compute_rbf_matrix_parallel(X, gamma, threshold=0.005, chunk_size=100):
    """Tính ma trận RBF Kernel song song với đầu ra là ma trận thưa."""
    n, d = X.shape
    W_data = []
    W_row = []
    W_col = []
    
    def compute_chunk(i_start, i_end):
        local_data = []
        local_row = []
        local_col = []
        for i in range(i_start, i_end):
            for j in range(n):
                dist = np.sum((X[i] - X[j]) ** 2)
                value = np.exp(-gamma * dist)
                if value > threshold:
                    local_data.append(value)
                    local_row.append(i)
                    local_col.append(j)
        return local_data, local_row, local_col
    
    chunks = [(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]
    results = Parallel(n_jobs=-1)(delayed(compute_chunk)(i, j) for i, j in chunks)
    
    for data, row, col in results:
        W_data.extend(data)
        W_row.extend(row)
        W_col.extend(col)
    
    W = csr_matrix((W_data, (W_row, W_col)), shape=(n, n))
    return W

# Tính ma trận trọng số song song với đầu ra là ma trận thưa
def compute_weight_matrix(image, sigma_i=0.3, sigma_x=20.0, chunk_size=100):
    h, w, c = image.shape
    coords = np.array(np.meshgrid(np.arange(h), np.arange(w))).reshape(2, -1).T
    features = image.reshape(-1, c)
    
    gamma_i = 1 / (2 * sigma_i**2)
    gamma_x = 1 / (2 * sigma_x**2)
    
    start_features = time.time()
    W_features = compute_rbf_matrix_parallel(features, gamma_i, threshold=0.005, chunk_size=chunk_size)
    end_features = time.time()
    W_features_time = end_features - start_features
    
    start_coords = time.time()
    W_coords = compute_rbf_matrix_parallel(coords, gamma_x, threshold=0.005, chunk_size=chunk_size)
    end_coords = time.time()
    W_coords_time = end_coords - start_coords
    
    W = W_features.multiply(W_coords)
    return W, W_features_time, W_coords_time

# Tính ma trận Laplacian với ma trận thưa
def compute_laplacian(W):
    D_diag = W.sum(axis=1).A1
    D = diags(D_diag)
    L = D - W
    return L, D

# Hàm tích vô hướng với Numba
@jit(nopython=True)
def handle_dot(a, b):
    result = 0.0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result

# Nhân ma trận-vector với Numba
@jit(nopython=True, parallel=True)
def matrix_vector_product(A_data, A_indices, A_indptr, v, n):
    result = np.zeros(n)
    for i in prange(n):
        for j in range(A_indptr[i], A_indptr[i + 1]):
            result[i] += A_data[j] * v[A_indices[j]]
    return result

# Thuật toán Lanczos cho ma trận thưa
@jit(nopython=True, parallel=True)
def lanczos_sparse(A_data, A_indices, A_indptr, v, m, n):
    V = np.zeros((m, n))
    T = np.zeros((m, m))
    V[0, :] = v / np.linalg.norm(v)
    
    w = matrix_vector_product(A_data, A_indices, A_indptr, V[0, :], n)
    alpha = handle_dot(w, V[0, :])
    w = w - alpha * V[0, :]
    T[0, 0] = alpha
    
    zero_vector = np.zeros(n, dtype=np.float64)
    for j in prange(1, m):
        beta = np.linalg.norm(w)
        if beta < 1e-10:
            break
        V[j, :] = w / beta
        for i in range(j):
            proj = handle_dot(V[j, :], V[i, :])
            V[j, :] -= proj * V[i, :]
        norm_vj = np.linalg.norm(V[j, :])
        if norm_vj < 1e-10:
            break
        V[j, :] /= norm_vj
        
        w = matrix_vector_product(A_data, A_indices, A_indptr, V[j, :], n)
        alpha = handle_dot(w, V[j, :])
        w = w - alpha * V[j, :] - (beta * V[j-1, :] if j > 0 else zero_vector)
    
        T[j, j] = alpha
        if j > 0:
            T[j-1, j] = beta
            T[j, j-1] = beta
    
    T = (T + T.T) / 2
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
    L_csr = L_normalized.tocsr()
    start_lanczos = time.time()
    T, V = lanczos_sparse(L_csr.data, L_csr.indices, L_csr.indptr, v0, m=k + 50, n=n)
    end_lanczos = time.time()
    
    lanczos_time = end_lanczos - start_lanczos
    j = T.shape[0] - 1
    if j < k:
        raise ValueError(f"Số lần lặp thực tế ({j}) nhỏ hơn k ({k})")
    
    eigvals, eigvecs_T = np.linalg.eig(T)
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs_T = eigvecs_T[:, idx]
    
    eigvecs_T = eigvecs_T[:, 1:k+1]
    eigvecs_original = D_inv_sqrt @ (V[:j+1, :].T @ eigvecs_T)
    
    for i in range(eigvecs_original.shape[1]):
        norm = np.linalg.norm(eigvecs_original[:, i])
        if norm > 1e-10:
            eigvecs_original[:, i] /= norm
        else:
            eigvecs_original[:, i] = np.zeros_like(eigvecs_original[:, i])
    
    return eigvecs_original, lanczos_time

# Gán nhãn
def assign_labels(eigen_vectors, k):
    return KMeans(n_clusters=k, random_state=42, n_init=50).fit(eigen_vectors).labels_

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

# Hàm chính xử lý ảnh 50x50 (không chia nhỏ thêm)
def normalized_cuts(lan, imagename, image_path, output_path):
    start_cpu = time.time()
    image = io.imread(image_path)
    image = color.gray2rgb(image) if image.ndim == 2 else image[:, :, :3] if image.shape[2] == 4 else image
    image = image / 255.0

    # Kiểm tra kích thước ảnh (phải là 50x50)
    if image.shape[0] != 50 or image.shape[1] != 50:
        print(f"❌ Ảnh {image_path} không có kích thước 50x50!")
        return None, None, None, None, None

    k = 2  # Số phân vùng

    # Tính toán ma trận trọng số và phân đoạn cho ảnh 50x50
    W, W_f, W_c = compute_weight_matrix(image, sigma_i=0.3, sigma_x=20.0)
    W_all = W_f + W_c
    L, D = compute_laplacian(W)
    vecs, lanczos_time = compute_eigen(L, D, k)
    labels = assign_labels(vecs, k)

    # Lưu file .seg cho ảnh
    seg_output_path = f"{imagename}_segmentation_{lan}.seg"
    save_seg_file(labels, image.shape, seg_output_path, imagename)

    end_cpu = time.time()
    total_cpu_time = end_cpu - start_cpu
    return total_cpu_time, [W_f], [W_c], [W_all], [lanczos_time]

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
        total_time, wf_times, wc_times, w_all_times, lanczos_times = normalized_cuts(i, imagename, image_path, output_excel_base)

        # Kiểm tra nếu không có kết quả trả về
        if total_time is None:
            continue

        # Ghi kết quả
        results.append([
            i, idx, file_name, 
            wf_times[0], wc_times[0], w_all_times[0], lanczos_times[0]
        ])

    df = pd.DataFrame(results, columns=[
        "Lần chạy", "Ảnh số", "Tên ảnh", 
        "Thời gian W đặc trưng (s)", "Thời gian W tọa độ (s)", "Thời gian W All", "Thời gian Lanczos (s)"
    ])
    output_excel = f"{output_excel_base}_{name}_{i}.xlsx"
    df.to_excel(output_excel, index=False, engine='openpyxl')
    print(f"✅ Kết quả Excel đã lưu vào {output_excel}")