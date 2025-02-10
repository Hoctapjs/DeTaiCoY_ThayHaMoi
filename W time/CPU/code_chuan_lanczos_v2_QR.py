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


# logging.basicConfig(level=logging.INFO)  

""" def kiemThuChayNhieuLan(i, name):
        temp_chuoi = f"{name}{i}"
        temp_chuoi = temp_chuoi + '.txt'
        logging.basicConfig(filename = temp_chuoi, level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
        # Duong dan toi anh cua ban
        image_path = "apple3_60x60.jpg"  # Thay bang duong dan anh cua ban
        image_path = "apple4_98x100.jpg"  # Thay bang duong dan anh cua ban
        normalized_cuts(image_path, k=3) """
        
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
    

    
    # Tinh do tuong dong ve dac trung va khong gian
    W_features = rbf_kernel(features, gamma=1/(2 * sigma_i**2))
    W_coords = rbf_kernel(coords, gamma=1/(2 * sigma_x**2))
    W = W_features * W_coords
    

    # Chuyen ma tran W sang dang ma tran thua COO
    W_sparse = coo_matrix(W)
    
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

#Version 4 - không dùng eig mà code lanzcos thuần - dùng thuật toán QR
def Lanczos(A, v, m):
    """
    Thuật toán Lanczos để xấp xỉ trị riêng và vector riêng.
    : A: Ma trận cần tính (numpy 2D array).
    : v: Vector khởi tạo.
    : m: Số bước lặp Lanczos.
    :return: Ma trận tam giác T và ma trận trực giao V.
    """
    n = len(v) # Đây là số phần tử trong vector v (số chiều của ma trận A)
    V = np.zeros((m, n)) # đây là một ma trận mxn lưu trữ các vector trực giao (là 2 vector có tích vô hướng = 0), mỗi hàng là một bước đã đi qua, np.zeros nghĩa là ban đầu tất cả các bước đi (hay các phần tử của ma trận) đều là 0, chưa đi bước nào
    T = np.zeros((m, m)) # đây là ma trận tam giác T
    V[0, :] = v / np.linalg.norm(v) # np.linalg.norm(v) là để tính chuẩn (độ dài) của vector = căn(v1^2 + v2^2 + ...)
    # => V[0, :] = v / np.linalg.norm(v) là để chuẩn hóa vector v đầu vào thành vector đơn vị 
    
    # Đoạn này là để làm cho w trực giao với V0 thôi
    # vd: để làm cho 2 vector a và b trực giao với nhau
    # 1. tính tích vô hướng của a và b (alpha)
    # 2. cập nhật vector a lại 
    #   a = a - alpha * b (b ở đây là V[0, :] = v / căn(v) )


    w = A @ V[0, :] # tính vector w bằng cách nhân A với vector đầu tiên của V - hiểu nôm na là w sẽ cho ta biết các mà ma trận A tương tác với vector khởi tạo v
    alpha = np.dot(w, V[0, :]) # .dot là tính tích vô hướng của 2 vector a và b (trong case này là w và vector đầu tiên của V), hệ số alpha là để đo mức độ song song giữa w và V0
    w = w - alpha * V[0, :]
    # alpha * V[0, :] tạo ra một vector có hướng song song với 
    # V[0,:] mà có độ dài tương ứng.
    # sau khi trừ xong thì nò sẽ loại bỏ phần song song ra khỏi w

    
    T[0, 0] = alpha # Gán giá trị alpha vào phần tử đầu tiên của T
    
    for j in range(1, m):
        beta = np.linalg.norm(w)
        if beta < 1e-10:
            break
        V[j, :] = w / beta
        w = A @ V[j, :]
        alpha = np.dot(w, V[j, :])
        w = w - alpha * V[j, :] - beta * V[j-1, :]
        
        T[j, j] = alpha
        T[j-1, j] = beta
        T[j, j-1] = beta
    
    return T, V

def QR_algorithm(T, max_iter=100, tol=1e-10):
    """
    Phương pháp QR để tính trị riêng và vector riêng của ma trận T.
    """
    n = T.shape[0]
    Q_total = np.eye(n)
    T_k = np.copy(T)
    
    for _ in range(max_iter):
        Q, R = np.linalg.qr(T_k)  # Phân rã QR
        T_k = R @ Q  # Lặp QR
        Q_total = Q_total @ Q  # Tích lũy Q để tìm vector riêng
        
        # Kiểm tra hội tụ
        if np.linalg.norm(np.triu(T_k, k=1)) < tol:
            break
    
    eigvals = np.diag(T_k)  # Trị riêng là các phần tử trên đường chéo
    eigvecs = Q_total  # Vector riêng là Q tổng hợp
    return eigvals, eigvecs

def compute_eigen(L, D, k=2):
    """
    Giải bài toán trị riêng bằng thuật toán Lanczos không dùng eigsh.
    """
    D_diag = D.diagonal().copy()
    D_diag[D_diag < 1e-10] = 1e-10
    D_inv_sqrt = diags(1.0 / np.sqrt(D_diag)).toarray()
    L_normalized = D_inv_sqrt @ L @ D_inv_sqrt
    
    v0 = np.random.rand(L.shape[0])
    v0 /= np.linalg.norm(v0)
    
    T, V = Lanczos(L_normalized, v0, m=k+5)
    
    eigvals, eigvecs_T = QR_algorithm(T[:k, :k])
    eigvecs_original = D_inv_sqrt @ (V[:k, :].T @ eigvecs_T)
    
    return eigvecs_original


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
    logging.info(f"Thoi gian COO: {end_cpu_coo - start_cpu_coo} giay")
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

