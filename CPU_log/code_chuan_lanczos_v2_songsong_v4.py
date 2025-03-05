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
from joblib import Parallel, delayed  

# logging.basicConfig(level=logging.INFO)  

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
    W_sparse = coo_matrix(W)
    logging.info(f"Kich thuoc ma tran thua (COO): {W_sparse.shape}")
    logging.info(f"Mau cua ma tran thua (COO) [du lieu, hang, cot]:\n{W_sparse.data[:9]}, {W_sparse.row[:9]}, {W_sparse.col[:9]}")
    
    return W_sparse



# 2. Tinh ma tran Laplace
def compute_laplacian(W_sparse):
    # Tạo ma trận đường chéo từ tổng các hàng
    D_diag = W_sparse.sum(axis=1).A.flatten() if hasattr(W_sparse, 'toarray') else W_sparse.sum(axis=1)
    D = np.diag(D_diag)  # Ma trận đường chéo
    L = D - W_sparse.toarray() if hasattr(W_sparse, 'toarray') else D -W_sparse  # Đảm bảo W là dạng mảng NumPy


    # Tạo ma trận đường chéo từ tổng các hàng
    # D_diag = W_sparse.sum(axis=1).A.flatten() 
    # D = np.diag(D_diag)  # Ma trận đường chéo
    # L = D - W_sparse # L = D - W

    logging.info("Kich thuoc ma tran duong cheo (D): %s", D.shape)
    logging.info("Mau cua D (9 phan tu dau):\n%s", D_diag[:9])  # In phần tử trên đường chéo
    logging.info("Kich thuoc ma tran Laplace (L): %s", L.shape)
    logging.info("Mau cua L (9x9 phan tu dau):\n%s", L[:9, :9])

    return L, D

# def compute_laplacian(W_sparse):
#     D_diag = np.array(W_sparse.sum(axis=1)).flatten()
#     D_inv_sqrt = diags(1.0 / np.sqrt(D_diag + 1e-10))  # Tránh chia cho 0
#     L_normalized = D_inv_sqrt @ (diags(D_diag) - W_sparse) @ D_inv_sqrt
#     return L_normalized, D_inv_sqrt

# def compute_eigen(L_normalized, k=2):
#     eigvals, eigvecs = eigsh(L_normalized, k=k, which='SM')
#     return eigvecs



# 3. Giai bai toan tri rieng

def handle_dot(a, b):  
    """Tính tích vô hướng của hai vector song song hóa"""  
    return np.sum(a * b)  

def matrix_vector_product(A, v):  
    """Hàm nhân ma trận với vector"""  
    return A @ v  

def Lanczos(A, v, m):  
    n = len(v)  
    V = np.zeros((m, n))  
    T = np.zeros((m, m))  
    V[0, :] = v / np.linalg.norm(v)  

    start = time.time()  
    
    # Sử dụng joblib để tính w  
    try:  
        w = Parallel(n_jobs=-1)(delayed(matrix_vector_product)(A, V[0, :]) for _ in range(1))[0]  # Nhân ma trận A với V[0, :]  
        # w = A @ V[0, :]
    except Exception as e:  
        logging.error(f"Error in matrix-vector product: {e}")  
        raise  

    end = time.time()  
    logging.info(f"Thoi gian nhan ma tran - vector (song song): {end - start:.6f} giay")  
    
    start = time.time()  

    # Sử dụng joblib để tính alpha  
    try:  
        alpha = Parallel(n_jobs=-1)(delayed(handle_dot)(w, V[0, :]) for _ in range(1))[0]  # Tính tích vô hướng song song  
    except Exception as e:  
        logging.error(f"Error in handle_dot: {e}")  
        raise  

    end = time.time()  
    logging.info(f"Thoi gian tinh tich vo huong (song song): {end - start:.6f} giay")  

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

    # start  
    # Áp dụng thuật toán Lanczos  
    T, V = Lanczos(L_normalized, v0, m=k+5)  # Sử dụng m > k để tăng độ chính xác  
    # end  
    
    # Tính trị riêng và vector riêng của ma trận tam giác T  
    eigvals, eigvecs_T = np.linalg.eig(T[:k, :k])  

    # Chuyển đổi vector riêng về không gian gốc  
    eigvecs_original = D_inv_sqrt @ (V[:k, :].T @ eigvecs_T)  

    return eigvecs_original  

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

