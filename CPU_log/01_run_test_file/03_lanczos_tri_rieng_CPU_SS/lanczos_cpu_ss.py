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
from joblib import Parallel, delayed  
from numba import njit, prange



# logging.basicConfig(level=logging.INFO)  
        
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
        _, lanczos_time   = normalized_cuts(i, file_name, image_path, output_excel)  # Bỏ time_w vì không cần
        
        # Lưu kết quả vào danh sách
        results.append([i, idx, file_name, lanczos_time])

    # Ghi kết quả vào file Excel
    df = pd.DataFrame(results, columns=["Lần chạy", "Ảnh số", "Tên ảnh", "Thời gian tổng Lanczos (s)"])

    # Tạo tên file kết quả theo format chuẩn
    output_excel = f"result_{name}_{i}.xlsx"
    
    df.to_excel(output_excel, index=False, engine='openpyxl')
    print(f"✅ Kết quả đã lưu vào {output_excel}")

# 1. Tinh ma tran trong so
# def compute_weight_matrix(image, sigma_i=0.1, sigma_x=10):
#     h, w, c = image.shape
#     coords = np.array(np.meshgrid(range(h), range(w))).reshape(2, -1).T  # Toa do (x, y)
#     features = image.reshape(-1, c)  # Dac trung mau
    
#     logging.info(f"Kich thuoc anh: {h}x{w}x{c}")
#     logging.info(f"Kich thuoc dac trung mau: {features.shape}, Kich thuoc toa do: {coords.shape}")
#     logging.info(f"Dac trung mau:\n{features[:9, :9]}")
#     logging.info(f"Toa do:\n{coords[:9, :9]}")

    
#     # Tinh do tuong dong ve dac trung va khong gian
#     W_features = rbf_kernel(features, gamma=1/(2 * sigma_i**2))
#     W_coords = rbf_kernel(coords, gamma=1/(2 * sigma_x**2))
#     W = W_features * W_coords
    
#     logging.info(f"Kich thuoc ma tran trong so (W): {W.shape}")
#     logging.info(f"Kich thuoc ma tran dac trung mau: {W_features.shape}, Kich thuoc ma tran toa do: {W_coords.shape}")
#     logging.info(f"Mau cua W_features (9x9 phan tu dau):\n{W_features[:9, :9]}")
#     logging.info(f"Mau cua W_coords (9x9 phan tu dau):\n{W_coords[:9, :9]}")
#     logging.info(f"Mau cua W (9x9 phan tu dau):\n{W[:9, :9]}")

#     # Chuyen ma tran W sang dang ma tran thua COO
#     W_sparse = coo_matrix(W)
#     logging.info(f"Kich thuoc ma tran thua (COO): {W_sparse.shape}")
#     logging.info(f"Mau cua ma tran thua (COO) [du lieu, hang, cot]:\n{W_sparse.data[:9]}, {W_sparse.row[:9]}, {W_sparse.col[:9]}")
    
#     return W_sparse

def compute_weight_matrix(image, sigma_i=0.1, sigma_x=10):
    h, w, c = image.shape
    coords = np.array(np.meshgrid(range(h), range(w))).reshape(2, -1).T  # Toa do (x, y)
    features = image.reshape(-1, c)  # Dac trung mau
    
    # Tinh do tuong dong ve dac trung va khong gian
    W_features = rbf_kernel(features, gamma=1/(2 * sigma_i**2))
    W_coords = rbf_kernel(coords, gamma=1/(2 * sigma_x**2))
    W = W_features * W_coords
    
    return W


# 2. Tinh ma tran Laplace
def compute_laplacian(W_sparse):
    # Tạo ma trận đường chéo từ tổng các hàng
    D_diag = W_sparse.sum(axis=1).A.flatten() if hasattr(W_sparse, 'toarray') else W_sparse.sum(axis=1)
    D = np.diag(D_diag)  # Ma trận đường chéo
    L = D - W_sparse.toarray() if hasattr(W_sparse, 'toarray') else D -W_sparse  # Đảm bảo W là dạng mảng NumPy

    logging.info("Kich thuoc ma tran duong cheo (D): %s", D.shape)
    logging.info("Mau cua D (9 phan tu dau):\n%s", D_diag[:9])  # In phần tử trên đường chéo
    logging.info("Kich thuoc ma tran Laplace (L): %s", L.shape)
    logging.info("Mau cua L (9x9 phan tu dau):\n%s", L[:9, :9])

    return L, D


# 3. Giai bai toan tri rieng
def handle_dot(a, b):
    """Tính tích vô hướng của hai vector song song hóa"""
    return np.sum(a * b)  # NumPy đã tối ưu hóa, nhưng có thể dùng joblib nếu cần

def matrix_vector_product(A, v):  
    """Hàm nhân ma trận với vector"""  
    return A @ v  

# Các phép toán A @ v và np.dot(v1, v2) vốn đã nhanh nếu NumPy sử dụng BLAS/MKL đa luồng. Nếu bạn muốn tận dụng đa lõi CPU, chỉ cần bật hỗ trợ OpenBLAS/MKL:
os.environ["OMP_NUM_THREADS"] = "2"  # Điều chỉnh số luồng tùy theo CPU 
# Các phép toán A @ v và np.dot(v1, v2) vốn đã nhanh nếu NumPy sử dụng BLAS/MKL đa luồng. Nếu muốn tận dụng đa lõi CPU, chỉ cần bật hỗ trợ OpenBLAS/MKL:
os.environ["OMP_NUM_THREADS"] = "6"  # Điều chỉnh số luồng tùy theo CPU 
# os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())  # Lấy toàn bộ số luồng CPU khả dụng

# from numba import njit, prange đoạn này import ở trên
# @njit(parallel=True) # version này thì thời gian khúc đầu chạy còn nhiều, và còn dùng break có thể gây lỗi
# def Lanczos(A, v, m):  
#     n = len(v)  
#     V = np.zeros((m, n))  
#     T = np.zeros((m, m))  
#     V[0, :] = v / np.linalg.norm(v)  

#     w = A @ V[0, :]  
#     alpha = np.dot(w, V[0, :])  
#     w = w - alpha * V[0, :]  
#     T[0, 0] = alpha  

#     for j in prange(1, m):  # prange để chạy song song
#         beta = np.linalg.norm(w)  
#         if beta < 1e-10:  
#             break  
#         V[j, :] = w / beta  
#         w = A @ V[j, :]  
#         alpha = np.dot(w, V[j, :])  
#         w = w - alpha * V[j, :] - beta * V[j-1, :]  

#         T[j, j] = alpha  
#         T[j-1, j] = beta  
#         T[j, j-1] = beta  

#     return T, V  

@njit(parallel=True, cache=True)  # Cache giúp giảm thời gian biên dịch lần đầu (thật ra nó chỉ ko báo lỗi cảnh báo như ở version ở trên ko có cache=True thôi chứ lần 1 vẫn lâu)
def Lanczos(A, v, m): # version mới
    n = len(v)
    V = np.zeros((m, n))
    T = np.zeros((m, m))
    V[0, :] = v / np.linalg.norm(v)

    for j in prange(1, m):  
        beta = np.linalg.norm(v)
        if beta < 1e-10:
            continue  # Có thể gây lỗi, nên thay bằng return hoặc continue
        
        V[j, :] = v / beta
        v = A @ V[j, :]
        alpha = np.dot(v, V[j, :])  # Tích vô hướng chuẩn hơn

        T[j, j] = alpha
        T[j-1, j] = beta
        T[j, j-1] = beta

    return T, V

# version cũ
# def Lanczos(A, v, m):  
#     n = len(v)  
#     V = np.zeros((m, n))  
#     T = np.zeros((m, m))  
#     V[0, :] = v / np.linalg.norm(v)  

#     # Sử dụng joblib để tính w  
#     try:  
#         w = Parallel(n_jobs=-1)(delayed(matrix_vector_product)(A, V[0, :]) for _ in range(1))[0]  # Nhân ma trận A với V[0, :]  
#     except Exception as e:  
#         logging.error(f"Error in matrix-vector product: {e}")  
#         raise  

#     # Sử dụng joblib để tính alpha  
#     try:  
#         alpha = Parallel(n_jobs=-1)(delayed(handle_dot)(w, V[0, :]) for _ in range(1))[0]  # Tính tích vô hướng song song  
#     except Exception as e:  
#         logging.error(f"Error in handle_dot: {e}")  
#         raise  

#     w = w - alpha * V[0, :]  
#     T[0, 0] = alpha  

#     for j in range(1, m):  
#         beta = np.linalg.norm(w)  
#         if beta < 1e-10:  
#             break  
#         V[j, :] = w / beta  
#         w = A @ V[j, :]  
#         alpha = handle_dot(w, V[j, :])  
#         w = w - alpha * V[j, :] - beta * V[j-1, :]  

#         T[j, j] = alpha  
#         T[j-1, j] = beta  
#         T[j, j-1] = beta  

#     return T, V  

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
    logging.info(f"Thoi gian lanczos khong song song(khong co COO): {lanczos_time_end - lanczos_time_start:.6f} giay")
    
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

def normalized_cuts(lan, imagename, image_path, output_path):
    start_cpu = time.time()  # Đo tổng thời gian xử lý
    logging.info(f"file name: {imagename}")
    logging.info(f"Lan thu: {lan}")

    # Đọc ảnh và chuẩn hóa
    image = io.imread(image_path)
    image = color.gray2rgb(image) if image.ndim == 2 else image[:, :, :3] if image.shape[2] == 4 else image
    image = image / 255.0
    k = 2  

    # Tính ma trận trọng số
    W = compute_weight_matrix(image)

    # Tính toán Laplacian
    L, D = compute_laplacian(W)

    # Giải eigen và đo thời gian Lanczos
    vecs, lanczos_time = compute_eigen(L, D, k)

    # Gán nhãn
    labels = assign_labels(vecs, k)

    # Lưu kết quả
    save_segmentation(image, labels, k, output_path)

    end_cpu = time.time()
    
    # Tính tổng thời gian
    total_cpu_time = end_cpu - start_cpu

    logging.info(f"⏳ Tổng thời gian: {total_cpu_time:.6f} giây")
    logging.info(f"⏳ Thời gian Lanczos: {lanczos_time:.6f} giây")

    return total_cpu_time, lanczos_time


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

