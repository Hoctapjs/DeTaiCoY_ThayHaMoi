import numpy as np
from sklearn.cluster import KMeans
from skimage import io, color
import time
import os
import pandas as pd
from datetime import datetime
import logging
from scipy.sparse import lil_matrix, csr_matrix, save_npz, load_npz, diags
import tkinter as tk
from tkinter import filedialog

# Hàm tính ma trận RBF thưa và lưu vào file
def compute_rbf_matrix_sparse(X, gamma, output_file="W_sparse.npz", threshold=1e-5):
    """Tính ma trận RBF thưa và lưu vào file thay vì giữ trong RAM."""
    n, d = X.shape
    W = lil_matrix((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(n):
            dist = np.sum((X[i] - X[j])**2)
            val = np.exp(-gamma * dist)
            if val > threshold:
                W[i, j] = val
    
    W_csr = W.tocsr()
    save_npz(output_file, W_csr)
    return output_file

# Tính ma trận trọng số và lưu vào file
def compute_weight_matrix(image, sigma_i=0.5, sigma_x=15.0, threshold=1e-5):
    h, w, c = image.shape
    coords = np.array(np.meshgrid(np.arange(h), np.arange(w))).reshape(2, -1).T
    features = image.reshape(-1, c)

    gamma_i = 1 / (2 * sigma_i**2)
    gamma_x = 1 / (2 * sigma_x**2)

    start_features = time.time()
    W_features_file = compute_rbf_matrix_sparse(features, gamma_i, "W_features.npz", threshold)
    end_features = time.time()
    W_features_time = end_features - start_features

    start_coords = time.time()
    W_coords_file = compute_rbf_matrix_sparse(coords, gamma_x, "W_coords.npz", threshold)
    end_coords = time.time()
    W_coords_time = end_coords - start_coords

    W_features = load_npz(W_features_file)
    W_coords = load_npz(W_coords_file)
    W = W_features.multiply(W_coords)
    W_file = "W_final.npz"
    save_npz(W_file, W)
    
    return W_file, W_features_time, W_coords_time

# Tính ma trận Laplacian từ file
def compute_laplacian(W_file):
    W = load_npz(W_file)
    D_diag = W.sum(axis=1).A1
    D = diags(D_diag)
    L = D - W
    L_file = "L.npz"
    save_npz(L_file, L)
    return L_file, D

# Hàm nhân ma trận-vector từ file
def matrix_vector_product_from_file(matrix_file, v):
    A = load_npz(matrix_file)
    return A.dot(v)

# Thuật toán Lanczos với ma trận từ file
def Lanczos_from_file(matrix_file, v, m):
    n = len(v)
    V = np.zeros((m, n))
    T = np.zeros((m, m))
    V[0, :] = v / np.linalg.norm(v)
    
    w = matrix_vector_product_from_file(matrix_file, V[0, :])
    alpha = np.dot(w, V[0, :])
    w = w - alpha * V[0, :]
    T[0, 0] = alpha
    
    for j in range(1, m):
        beta = np.linalg.norm(w)
        if beta < 1e-10:
            break
        V[j, :] = w / beta
        for i in range(j):
            proj = np.dot(V[j, :], V[i, :])
            V[j, :] -= proj * V[i, :]
        norm_vj = np.linalg.norm(V[j, :])
        if norm_vj < 1e-10:
            break
        V[j, :] /= norm_vj
        
        w = matrix_vector_product_from_file(matrix_file, V[j, :])
        alpha = np.dot(w, V[j, :])
        w = w - alpha * V[j, :] - (beta * V[j-1, :] if j > 0 else 0)
        
        T[j, j] = alpha
        if j > 0:
            T[j-1, j] = beta
            T[j, j-1] = beta
    
    T = (T + T.T) / 2
    return T[:j+1, :j+1], V[:j+1, :]

# Tính trị riêng và vector riêng từ file
def compute_eigen(L_file, D, k=2):
    D_diag = D.diagonal().copy()
    D_diag[D_diag < 1e-10] = 1e-10
    D_inv_sqrt = diags(1.0 / np.sqrt(D_diag))
    
    v0 = np.random.rand(D.shape[0])
    v0 /= np.linalg.norm(v0)

    lanczos_time_start = time.time()
    m = k + 50
    T, V = Lanczos_from_file(L_file, v0, m)
    lanczos_time_end = time.time()
    lanczos_time = lanczos_time_end - lanczos_time_start
    
    eigvals, eigvecs_T = np.linalg.eigh(T)
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs_T = eigvecs_T[:, idx]
    
    eigvecs_T = eigvecs_T[:, 1:k+1]
    eigvecs_original = D_inv_sqrt @ (V.T @ eigvecs_T)
    
    for i in range(eigvecs_original.shape[1]):
        norm = np.linalg.norm(eigvecs_original[:, i])
        if norm > 1e-10:
            eigvecs_original[:, i] /= norm
    
    logging.info(f"Thời gian Lanczos: {lanczos_time:.6f} giây")
    logging.info(f"Trị riêng: {eigvals[:k+1]}")
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
    k = 2

    W_file, W_f, W_c = compute_weight_matrix(image, sigma_i=0.5, sigma_x=15.0)
    W_all = W_f + W_c
    L_file, D = compute_laplacian(W_file)
    vecs, lanczos_time = compute_eigen(L_file, D, k)
    labels = assign_labels(vecs, k)

    seg_output_path = f"{imagename}_segmentation_{lan}.seg"
    save_seg_file(labels, image.shape, seg_output_path, imagename)

    end_cpu = time.time()
    total_cpu_time = end_cpu - start_cpu
    
    for temp_file in ["W_features.npz", "W_coords.npz", "W_final.npz", "L.npz"]:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    return total_cpu_time, W_f, W_c, W_all

# Hàm chạy nhiều ảnh với hộp thoại chọn thư mục
def kiemThuChayNhieuLan(i, name, output_excel_base="results"):
    # Khởi tạo Tkinter và ẩn cửa sổ chính
    root = tk.Tk()
    root.withdraw()  # Ẩn cửa sổ Tkinter chính

    # Hiển thị hộp thoại chọn thư mục
    folder_path = filedialog.askdirectory(title="Chọn thư mục chứa ảnh")
    if not folder_path:  # Nếu người dùng không chọn thư mục
        print("❌ Không có thư mục nào được chọn!")
        root.destroy()
        return

    if not os.path.isdir(folder_path):
        print(f"❌ Thư mục {folder_path} không tồn tại!")
        root.destroy()
        return

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    if not image_files:
        print(f"❌ Không tìm thấy file ảnh nào trong {folder_path}!")
        root.destroy()
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

    root.destroy()  # Đóng Tkinter sau khi hoàn tất

# Chạy thử
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    kiemThuChayNhieuLan(1, "test")