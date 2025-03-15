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
from scipy.sparse import coo_matrix #chuyển sang ma trận coo
from scipy.sparse import isspmatrix, diags
import logging
import os
import pandas as pd
        
def kiemThuChayNhieuLan(i, name, folder_path, output_excel="results.xlsx"):
    # Kiểm tra thư mục
    if not os.path.isdir(folder_path):
        print(f"❌ Thư mục {folder_path} không tồn tại!")
        return
    
    # Lấy danh sách file ảnh
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    if not image_files:
        print(f"❌ Không tìm thấy file ảnh nào trong {folder_path}!")
        return

    results = []  # Danh sách lưu kết quả

    for idx, file_name in enumerate(image_files, start=1):
        image_path = os.path.join(folder_path, file_name)
        print(f"📷 Đang xử lý ảnh {idx}: {image_path}")
        
        # Xử lý ảnh
        _, wf_time, wc_time, W_all   = normalized_cuts(i, file_name, image_path, output_excel)  # Bỏ time_w vì không cần
        
        # Lưu kết quả vào danh sách
        results.append([i, idx, file_name, wf_time, wc_time, W_all])

    # Ghi kết quả vào file Excel
    df = pd.DataFrame(results, columns=["Lần chạy", "Ảnh số", "Tên ảnh", "Thời gian W đặc trưng (s)", "Thời gian W tọa độ (s)", "Thời gian W All"])


    # Tạo tên file kết quả theo format chuẩn
    output_excel = f"result_{name}_{i}.xlsx"
    
    df.to_excel(output_excel, index=False, engine='openpyxl')
    print(f"✅ Kết quả đã lưu vào {output_excel}")

import numpy as np
import time

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

def compute_weight_matrix(image, sigma_i=0.1, sigma_x=10):
    """Tính ma trận trọng số trên CPU bằng vòng lặp thuần Python."""
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



# def compute_laplacian(W):
#     D = np.diag(W.sum(axis=1))
#     L = D - W
#     return L, D

# from scipy.sparse import diags, issparse
# import numpy as np
# import time
# from scipy.sparse.linalg import eigsh

# def compute_eigen(L, D, k=2):
#     if issparse(D):
#         D = D.todense()  # Chuyển sang dạng dense nếu là sparse

#     D_diag = np.array(D.diagonal()).copy()  # Bản sao có thể chỉnh sửa
#     D_diag[D_diag < 1e-10] = 1e-10  # Tránh giá trị quá nhỏ gây lỗi

#     if issparse(D):
#         D.setdiag(D_diag)  # Cập nhật đường chéo nếu là sparse matrix
#     else:
#         np.fill_diagonal(D, D_diag)  # Nếu là numpy array, dùng fill_diagonal()

#     D_inv_sqrt = diags(1.0 / np.sqrt(D_diag))  # Tạo ma trận nghịch đảo căn
#     L_normalized = D_inv_sqrt @ L @ D_inv_sqrt  # Chuẩn hóa Laplacian

#     start_time = time.time()
#     eigvals, eigvecs = eigsh(L_normalized, k, which='SM')  # Tính eigen
#     end_time = time.time()

#     lanczos_time = end_time - start_time
#     eigvecs_original = D_inv_sqrt @ eigvecs  # Khôi phục eigen vectors gốc

#     return eigvecs_original, lanczos_time

# tạm thời khóa để test độ chính xác

# 2. Tinh ma tran Laplace
def compute_laplacian(W_sparse):
    # Tạo ma trận đường chéo từ tổng các hàng
    D_diag = W_sparse.sum(axis=1).A.flatten() if hasattr(W_sparse, 'toarray') else W_sparse.sum(axis=1)
    D = np.diag(D_diag)  # Ma trận đường chéo
    L = D - W_sparse.toarray() if hasattr(W_sparse, 'toarray') else D -W_sparse  # Đảm bảo W là dạng mảng NumPy

    return L, D


# 3. Giai bai toan tri rieng
def handle_dot(a, b):
    """Tính tích vô hướng của hai vector song song hóa"""
    return np.sum(a * b)  # NumPy đã tối ưu hóa, nhưng có thể dùng joblib nếu cần

def matrix_vector_product(A, v):  
    """Hàm nhân ma trận với vector"""  
    return A @ v  

def Lanczos(A, v, m):
    n = len(v)
    V = np.zeros((m, n))
    T = np.zeros((m, m))
    V[0, :] = v / np.linalg.norm(v)
    
    w = matrix_vector_product(A, V[0, :])
    alpha = handle_dot(w, V[0, :])  # Tích vô hướng song song
    
    w = w - alpha * V[0, :]
    T[0, 0] = alpha
    
    for j in range(1, m):
        beta = np.linalg.norm(w)
        if beta < 1e-10:
            break
        V[j, :] = w / beta
        w = A @ V[j, :]
        alpha = handle_dot(w, V[j, :])
        w = w - alpha * V[j, :] - beta * V[j-1, :]
        
        T[j, j] = alpha
        T[j-1, j] = beta
        T[j, j-1] = beta
    
    return T, V

def compute_eigen(L, D, k=2):
    """
    Giải bài toán trị riêng bằng thuật toán Lanczos không dùng eigsh.
    :param L: Ma trận Laplace thưa (Scipy sparse matrix).
    :param D: Ma trận đường chéo (Scipy sparse matrix).
    :param k: Số trị riêng nhỏ nhất cần tính.
    :return: Các vector riêng tương ứng (k vector).
    """
    # Chuan hoa ma tran Laplace: D^-1/2 * L * D^-1/2
    D_diag = D.diagonal().copy()  # Lay duong cheo cua D
    D_diag[D_diag < 1e-10] = 1e-10  # Tranh chia cho 0 hoac gan 0
    D_inv_sqrt = diags(1.0 / np.sqrt(D_diag))  # Tinh D^-1/2
    L_normalized = D_inv_sqrt @ L @ D_inv_sqrt  # Chuan hoa ma tran Laplace
    
    # Khởi tạo vector ngẫu nhiên
    v0 = np.random.rand(L.shape[0])
    v0 /= np.linalg.norm(v0)

    # Áp dụng thuật toán Lanczos
    lanczos_time_start = time.time()
    T, V = Lanczos(L_normalized, v0, m=k+5)  # Sử dụng m > k để tăng độ chính xác
    lanczos_time_end = time.time()

    # Thời gian Lanczos
    lanczos_time = lanczos_time_end - lanczos_time_start
    
    # Tính trị riêng và vector riêng của ma trận tam giác T
    eigvals, eigvecs_T = np.linalg.eig(T[:k, :k])
    
    # Chuyển đổi vector riêng về không gian gốc
    eigvecs_original = D_inv_sqrt @ (V[:k, :].T @ eigvecs_T)
    
    return eigvecs_original, lanczos_time

# 4. Gan nhan cho tung diem anh dua tren vector rieng
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

from datetime import datetime

def save_seg_file(labels, image_shape, output_path, image_name="image"):
    h, w = image_shape[:2]
    unique_labels = np.unique(labels)
    segments = len(unique_labels)
    
    # Tạo phần header
    header = [
        "format ascii cr",
        f"date {datetime.now().strftime('%a %b %d %H:%M:%S %Y')}",
        f"image {image_name}",
        "user 1102",  # Giữ nguyên như file mẫu
        f"width {w}",
        f"height {h}",
        f"segments {segments}",
        "gray 0",
        "invert 0",
        "flipflop 0",
        "data"
    ]
    
    # Tạo dữ liệu pixel theo định dạng (nhãn, dòng, cột bắt đầu, cột kết thúc)
    data_lines = []
    for row in range(h):
        row_labels = labels[row, :]
        start_col = 0
        current_label = row_labels[0]
        
        for col in range(1, w):
            if row_labels[col] != current_label:
                data_lines.append(f"{current_label} {row} {start_col} {col}")
                start_col = col
                current_label = row_labels[col]
        
        # Thêm dòng cuối cùng của hàng
        data_lines.append(f"{current_label} {row} {start_col} {w}")
    
    # Lưu vào file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n")
        f.write("\n".join(data_lines) + "\n")
    
    print(f"✅ File SEG đã lưu: {output_path}")

def normalized_cuts(lan, imagename, image_path, output_path):
    start_cpu = time.time()  # Đo tổng thời gian xử lý
    
    # Đọc ảnh và chuẩn hóa
    image = io.imread(image_path)
    image = color.gray2rgb(image) if image.ndim == 2 else image[:, :, :3] if image.shape[2] == 4 else image
    image = image / 255.0
    k = 2  

    # Tính ma trận trọng số
    W, W_f, W_c = compute_weight_matrix(image)
    W_all = W_f + W_c

    # Tính toán Laplacian
    L, D = compute_laplacian(W)

    # Giải eigen và đo thời gian Lanczos
    vecs, lanczos_time = compute_eigen(L, D, k)

    # Gán nhãn
    labels = assign_labels(vecs, k)

    # Lưu kết quả dưới dạng file SEG
    seg_output_path = f"{imagename}_segmentation.seg"
    save_seg_file(labels.reshape(image.shape[:2]), image.shape, seg_output_path, imagename)

    end_cpu = time.time()
    
    # Tính tổng thời gian
    total_cpu_time = end_cpu - start_cpu

    return total_cpu_time, W_f, W_c, W_all



# 7. Mo file chon anh tu hop thoai
def open_file_dialog():
    # Tao cua so an cho tkinter
    root = Tk()
    root.withdraw()  # An cua so chinh
    
    # Mo hop thoai chon file anh
    file_path = askopenfilename(title="Chon anh", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
    return file_path   

# 8. Chay thu nghiem
if __name__ == "__main__":
    # Mo hop thoai chon anh
    image_path = open_file_dialog()
    if image_path:
        logging.info(f"Da chon anh: {image_path}")
        normalized_cuts(image_path, k=3)  # Phan vung thanh 3 nhom
    else:
        logging.info("Khong co anh nao duoc chon.")

