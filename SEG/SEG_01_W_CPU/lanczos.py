import numpy as np
from sklearn.cluster import KMeans
from skimage import io, color
import time
import os
import pandas as pd
from datetime import datetime
import logging
from scipy.sparse import diags

# Hàm tính ma trận RBF (logic cũ - ma trận dày đặc, không ngưỡng)
def compute_rbf_matrix(X, gamma):
    """Tính ma trận RBF Kernel trên CPU bằng vòng lặp thuần Python."""
    n, d = X.shape
    W = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(n):
            dist = 0.0
            for k in range(d):
                diff = X[i, k] - X[j, k]
                dist += diff * diff
            W[i, j] = np.exp(-gamma * dist)

    return W

# Tính ma trận trọng số (logic cũ - ma trận dày đặc)
def compute_weight_matrix(image, sigma_i=0.5, sigma_x=15.0):  # Giữ nguyên tham số tối ưu
    h, w, c = image.shape
    coords = np.array(np.meshgrid(np.arange(h), np.arange(w))).reshape(2, -1).T  # Tọa độ (x, y)
    features = image.reshape(-1, c)  # Chuyển toàn bộ đặc trưng thành mảng 2D

    gamma_i = 1 / (2 * sigma_i**2)
    gamma_x = 1 / (2 * sigma_x**2)

    # Tính ma trận trọng số dựa trên đặc trưng màu
    start_features = time.time()
    W_features = compute_rbf_matrix(features, gamma_i)
    end_features = time.time()
    W_features_time = end_features - start_features

    # Tính ma trận trọng số dựa trên tọa độ không gian
    start_coords = time.time()
    W_coords = compute_rbf_matrix(coords, gamma_x)
    end_coords = time.time()
    W_coords_time = end_coords - start_coords

    # Nhân hai ma trận trọng số để tạo ma trận W cuối cùng
    W = np.multiply(W_features, W_coords)
    return W, W_features_time, W_coords_time

# Tính ma trận Laplacian
def compute_laplacian(W):
    D_diag = W.sum(axis=1)
    D = np.diag(D_diag)
    L = D - W
    return L, D

# Hàm tích vô hướng
def handle_dot(a, b):
    result = 0.0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result

# Hàm nhân ma trận-vector
def matrix_vector_product(A, v):
    n = A.shape[0]
    result = np.zeros(n)
    for i in range(n):
        for j in range(n):
            result[i] += A[i, j] * v[j]
    return result

# Thuật toán Lanczos với trực giao hóa chặt chẽ
def Lanczos(A, v, m):
    n = len(v)
    V = np.zeros((m, n))
    T = np.zeros((m, m))
    V[0, :] = v / np.linalg.norm(v)
    
    w = matrix_vector_product(A, V[0, :])
    alpha = handle_dot(w, V[0, :])
    w = w - alpha * V[0, :]
    T[0, 0] = alpha
    
    for j in range(1, m):
        beta = np.linalg.norm(w)
        if beta < 1e-10:
            break
        V[j, :] = w / beta
        # Trực giao hóa (Gram-Schmidt) chặt chẽ
        for i in range(j):
            proj = handle_dot(V[j, :], V[i, :])
            V[j, :] -= proj * V[i, :]
        norm_vj = np.linalg.norm(V[j, :])
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

# Tính trị riêng và vector riêng
def compute_eigen(L, D, k=2):
    D_diag = D.diagonal().copy()
    D_diag[D_diag < 1e-10] = 1e-10
    D_inv_sqrt = diags(1.0 / np.sqrt(D_diag))
    L_normalized = D_inv_sqrt @ L @ D_inv_sqrt
    
    v0 = np.random.rand(L.shape[0])
    v0 /= np.linalg.norm(v0)

    lanczos_time_start = time.time()
    m = k + 50  # Tăng số lần lặp để cải thiện độ chính xác
    T, V = Lanczos(L_normalized, v0, m)
    lanczos_time_end = time.time()

    lanczos_time = lanczos_time_end - lanczos_time_start
    logging.info(f"Thời gian Lanczos: {lanczos_time:.6f} giây")
    
    # Tính trị riêng và vector riêng của ma trận T
    eigvals, eigvecs_T = np.linalg.eig(T)
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs_T = eigvecs_T[:, idx]
    
    # Bỏ vector riêng nhỏ nhất (trị riêng gần 0)
    eigvecs_T = eigvecs_T[:, 1:k+1]
    eigvecs_original = D_inv_sqrt @ (V.T @ eigvecs_T)
    
    # Chuẩn hóa vector riêng
    for i in range(eigvecs_original.shape[1]):
        norm = np.linalg.norm(eigvecs_original[:, i])
        if norm > 1e-10:
            eigvecs_original[:, i] /= norm
    
    logging.info(f"Trị riêng: {eigvals[:k+1]}")
    logging.info(f"Vector riêng (mẫu): {eigvecs_original[:5, :]}")
    return eigvecs_original, lanczos_time

# Gán nhãn
def assign_labels(eigen_vectors, k):
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
    return total_cpu_time, W_f, W_c, W_all, lanczos_time  # Thêm lanczos_time vào kết quả trả về

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
        total_time, wf_time, wc_time, W_all, lanczos_time = normalized_cuts(i, imagename, image_path, output_excel_base)
        results.append([i, idx, file_name, wf_time, wc_time, W_all, lanczos_time])

    df = pd.DataFrame(results, columns=[
        "Lần chạy", 
        "Ảnh số", 
        "Tên ảnh", 
        "Thời gian W đặc trưng (s)", 
        "Thời gian W tọa độ (s)", 
        "Thời gian W All", 
        "Thời gian Lanczos (s)"  # Thêm cột mới
    ])
    output_excel = f"{output_excel_base}_{name}_{i}.xlsx"
    df.to_excel(output_excel, index=False, engine='openpyxl')
    print(f"✅ Kết quả Excel đã lưu vào {output_excel}")