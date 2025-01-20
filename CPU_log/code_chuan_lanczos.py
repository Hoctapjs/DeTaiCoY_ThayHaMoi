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
import logging
from scipy.sparse import coo_matrix #chuyển sang ma trận coo


logging.basicConfig(level=logging.INFO)  

def kiemThuChayNhieuLan(i, name):
        temp_chuoi = f"{name}{i}"
        temp_chuoi = temp_chuoi + '.txt'
        logging.basicConfig(filename = temp_chuoi, level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
        # Duong dan toi anh cua ban
        image_path = "apple3_60x60.jpg"  # Thay bang duong dan anh cua ban
        """ image_path = "apple4_98x100.jpg"  # Thay bang duong dan anh cua ban """
        normalized_cuts(image_path, k=3)
        


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

# def compute_laplacian(W_sparse):
#     # Tạo ma trận đường chéo từ tổng các hàng
#     D_diag = W_sparse.sum(axis=1).A.flatten() if hasattr(W_sparse, 'toarray') else W_sparse.sum(axis=1)
#     D = np.diag(D_diag)  # Ma trận đường chéo
#     L = D - W_sparse.toarray() if hasattr(W_sparse, 'toarray') else D -W_sparse  # Đảm bảo W là dạng mảng NumPy

#     logging.info("Kich thuoc ma tran duong cheo (D): %s", D.shape)
#     logging.info("Mau cua D (9 phan tu dau):\n%s", D_diag[:9])  # In phần tử trên đường chéo
#     logging.info("Kich thuoc ma tran Laplace (L): %s", L.shape)
#     logging.info("Mau cua L (9x9 phan tu dau):\n%s", L[:9, :9])

#     return L, D

def compute_laplacian(W_sparse):
    D_diag = np.array(W_sparse.sum(axis=1)).flatten()
    D_inv_sqrt = diags(1.0 / np.sqrt(D_diag + 1e-10))  # Tránh chia cho 0
    L_normalized = D_inv_sqrt @ (diags(D_diag) - W_sparse) @ D_inv_sqrt
    return L_normalized
def compute_eigen(L_normalized, k=2):
    eigvals, eigvecs = eigsh(L_normalized, k=k, which='SM')
    return eigvecs



# 3. Giai bai toan tri rieng
# def compute_eigen(L, D, k=2):
#     """
#     Giai bai toan tri rieng bang thuat toan Lanczos (eigsh) tren GPU.
#     :param L: Ma tran Laplace thua (CuPy sparse matrix).
#     :param D: Ma tran duong cheo (CuPy sparse matrix).
#     :param k: So tri rieng nho nhat can tinh.
#     :return: Cac vector rieng tuong ung (k vector).
#     """
#     # Chuan hoa ma tran Laplace: D^-1/2 * L * D^-1/2
#     D_diag = D.diagonal().copy()  # Lay duong cheo cua D
#     D_diag[D_diag < 1e-10] = 1e-10  # Trahn chia cho 0 hoac gan 0
#     D_inv_sqrt = diags(1.0 / np.sqrt(D_diag))  # Tinh D^-1/2
#     L_normalized = D_inv_sqrt @ L @ D_inv_sqrt  # Chuan hoa ma tran Laplace

#     # Giai bai toan tri rieng bang eigsh
#     eigvals, eigvecs = eigsh(L_normalized, k=k, which='SA')  # Dung SA thay vi SM

#     # Chuyen lai eigenvectors ve khong gian goc bang cach nhan D^-1/2
#     eigvecs_original = D_inv_sqrt @ eigvecs

#     return eigvecs_original


# def compute_eigen(W_sparse, k=2):  
#     """  
#     Tính trị riêng và vector riêng bằng thuật toán Lanczos.  
#     :param W_sparse: Ma trận trọng số dạng COO (đối xứng).  
#     :param k: Số trị riêng nhỏ nhất cần tính.  
#     :return: Các vector riêng tương ứng (k vector).  
#     """  
#     n = W_sparse.shape[0]  # Kích thước ma trận  

#     # 1. Khởi tạo biến  
#     v = np.zeros(n)  # Khởi tạo vector v  
#     beta = 1.0  # Khởi tạo β0  
#     k_iter = 0  # Số lần lặp k  
#     T = np.zeros((min(k, n), min(k, n)))  # Ma trận tridiagonal T  
#     V = np.zeros((n, min(k, n)))  # Ma trận chứa các vector Lanczos  
#     W = W_sparse.toarray()  # Chuyển ma trận vào dạng NumPy cho tính toán  

#     # 2. Gán giá trị cho vector đơn vị ban đầu  
#     v = np.random.rand(n)  
#     v /= np.linalg.norm(v)  

#     while beta > 1e-10 and k_iter < k:  
#         V[:, k_iter] = v  # Gán vector hiện tại vào V  
#         w = W @ v  # Tính Av  

#         if k_iter > 0:  
#             w -= beta * V[:, k_iter - 1]  # Thực hiện phép trừ β*v_(k-1)  

#         # Tính alpha = <v, w>  
#         alpha = np.dot(v, w)  
#         T[k_iter, k_iter] = alpha  # Gán giá trị alpha vào đường chéo của T  
#         w -= alpha * v  # Thực hiện phép trừ alpha*v  

#         # Tính beta = ||w||  
#         beta = np.linalg.norm(w)  
#         if beta > 1e-10:  
#             v = w / beta  # Chuẩn hóa v  

#         if k_iter < k - 1:  
#             T[k_iter, k_iter + 1] = beta  # Gán giá trị vào đường chéo trên  
#             T[k_iter + 1, k_iter] = beta  # Gán giá trị vào đường chéo dưới  

#         k_iter += 1  

#     # 3. Tính trị riêng và vector riêng từ ma trận T  
#     eigenvalues, eigenvectors = np.linalg.eigh(T)  # Tính toán trị riêng và vector riêng  

#     # 4. Chọn k trị riêng nhỏ nhất và vector riêng tương ứng  
#     idx = np.argsort(eigenvalues)[:k]  
#     eigenvalues = eigenvalues[idx]  
#     eigenvectors = eigenvectors[:, idx]  

#     # 5. Chuyển đổi vector riêng về không gian gốc  
#     eigvecs_original = V[:, :k] @ eigenvectors  # Chuyển đổi về không gian gốc  

#     return eigvecs_original  # Trả về vector riêng


# 4. Gan nhan cho tung diem anh dua tren vector rieng
def assign_labels(eigen_vectors, k):
    # Dung K-Means de gan nhan
    kmeans = KMeans(n_clusters=k, random_state=0).fit(eigen_vectors)
    labels = kmeans.labels_
    
    logging.info(f"Nhan gan cho 27 pixel dau tien: {labels[:27]}")
    return labels

# 5. Hien thi ket qua
def display_segmentation(image, labels, k):
    h, w, c = image.shape
    segmented_image = np.zeros_like(image, dtype=np.uint8)
    
    # Tao bang mau ngau nhien
    colors = np.random.randint(0, 255, size=(k, 3), dtype=np.uint8)
    
    # To mau tung vung
    for i in range(k):
        segmented_image[labels.reshape(h, w) == i] = colors[i]
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Anh goc")
    plt.imshow(image)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Anh sau phan vung")
    plt.imshow(segmented_image)
    plt.axis('off')
    plt.show()

# 6. Ket hop toan bo
def normalized_cuts(image_path, k):
    # Tinh thoi gian tren CPU
    start_cpu = time.time()

    # Doc anh va chuan hoa
    image = io.imread(image_path)
    if image.ndim == 2:  # Neu la anh xam, chuyen thanh RGB
        image = color.gray2rgb(image)
    elif image.shape[2] == 4:  # Neu la anh RGBA, loai bo kenh alpha
        image = image[:, :, :3]
    image = image / 255.0  # Chuan hoa ve [0, 1]
    
    # Tinh toan Ncuts
    logging.info("Bat dau tinh ma tran trong so...")
    W = compute_weight_matrix(image)
    
    logging.info("Tinh ma tran Laplace...")
    L, D = compute_laplacian(W)
    
    logging.info("Tinh vector rieng...")
    # eigen_vectors = compute_eigen(L, D, k=k)  # Tinh k vector rieng
    eigen_vectors = compute_eigen(W, k=k)  # Tinh k vector rieng
    
    logging.info("Gan nhan cho cac diem anh...")
    labels = assign_labels(eigen_vectors, k)  # Gan nhan cho moi diem anh
    
    logging.info("Hien thi ket qua...")

    end_cpu = time.time()
    logging.info(f"Thoi gian: {end_cpu - start_cpu} giay")

    display_segmentation(image, labels, k)

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

