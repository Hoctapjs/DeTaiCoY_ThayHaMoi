import cupy as cp  # Thay thế NumPy bằng CuPy
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
# from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from skimage import io, color
import cupyx.scipy.sparse as sp
from cupyx.scipy.sparse.linalg import eigsh
from cupyx.scipy.sparse import diags
import time
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import logging
from cupyx.scipy.sparse import coo_matrix


def kiemThuChayNhieuLan(i, name):
        temp_chuoi = f"{name}{i}"
        temp_chuoi = temp_chuoi + '.txt'
        logging.basicConfig(filename = temp_chuoi, level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
        # Duong dan toi anh cua ban
        image_path = "apple3_60x60.jpg"  # Thay bang duong dan anh cua ban
        """ image_path = "apple4_98x100.jpg"  # Thay bang duong dan anh cua ban """
        normalized_cuts(image_path, k=3)

        # Mở hộp thoại chọn ảnh
        # image_path = open_file_dialog()
        # if image_path:
        #     logging.info(f"Da chon anh: {image_path}")
        #     normalized_cuts(image_path, k=3)  # Phan vung thanh 3 nhom
        # else:
        #     logging.info("Khong co anh nao duoc chon.")


# 1. Tinh ma tran trong so
def compute_weight_matrix(image, sigma_i=0.1, sigma_x=10):
    h, w, c = image.shape
    coords = cp.array(cp.meshgrid(cp.arange(h), cp.arange(w))).reshape(2, -1).T  # Tọa độ (x, y)
    features = image.reshape(-1, c)  # Đặc trưng màu

    logging.info(f"Kich thuoc anh: {h}x{w}x{c}")
    logging.info(f"Kich thuoc dac trung mau: {features.shape}, Kich thuoc toa do: {coords.shape}")
    logging.info(f"Dac trung mau:\n{features[:9, :9]}")
    logging.info(f"Toa do:\n{coords[:9, :9]}")

    # Tính độ tương đồng về đặc trưng và không gian
    W_features = rbf_kernel(cp.asnumpy(features), gamma=1/(2 * sigma_i**2))  # Chuyển dữ liệu từ GPU sang CPU
    W_coords = rbf_kernel(cp.asnumpy(coords), gamma=1/(2 * sigma_x**2))  # Chuyển dữ liệu từ GPU sang CPU

    # Chuyển kết quả từ NumPy (CPU) sang CuPy (GPU)
    W_features = cp.asarray(W_features)
    W_coords = cp.asarray(W_coords)

    W = cp.multiply(W_features, W_coords)  # Phép nhân phần tử của ma trận trên GPU

    # Chuyển thành ma trận thưa dạng COO
    W_sparse = coo_matrix(W)

    logging.info(f"Kich thuoc ma tran trong so (W): {W.shape}")
    logging.info(f"Kich thuoc ma tran thua (W_sparse): {W_sparse.shape}")
    logging.info(f"So luong phan tu khac 0: {W_sparse.nnz}")
    logging.info(f"Mau cua W_features (9x9 phan tu dau):\n{W_features[:9, :9]}")
    logging.info(f"Mau cua W_coords (9x9 phan tu dau):\n{W_coords[:9, :9]}")
    logging.info(f"Mau cua W (9x9 phan tu dau):\n{W[:9, :9]}")

    return W_sparse


# 2. Tinh ma tran Laplace
def compute_laplacian(W_sparse):
    # Tổng của các hàng trong ma trận W
    D_diag = W_sparse.sum(axis=1).get().flatten()  # Tính tổng các hàng
    D = cp.diag(D_diag)  # Tạo ma trận đường chéo từ tổng hàng
    L = D - W_sparse  # L = D - W
    logging.info("Kich thuoc ma tran duong cheo (D): %s", D.shape)
    logging.info("Mau cua D (9 phan tu dau):\n%s", D_diag[:9])  # In phần tử trên đường chéo
    logging.info("Kich thuoc ma tran Laplace (L): %s", L.shape)
    logging.info("Mau cua L (9x9 phan tu dau):\n%s", L[:9, :9])
    return L, D

# 3. Giai bai toan tri rieng
# def lanczos_method(L, k, max_iter=100, tol=1e-6):
#     """
#     Triển khai thuật toán Lanczos để tìm trị riêng nhỏ nhất của ma trận L.
    
#     Args:
#         L (cupy.ndarray): Ma trận đối xứng cần tìm trị riêng.
#         k (int): Số trị riêng nhỏ nhất cần tìm.
#         max_iter (int): Số vòng lặp tối đa.
#         tol (float): Ngưỡng hội tụ.

#     Returns:
#         eigvals (cupy.ndarray): k trị riêng nhỏ nhất.
#         eigvecs (cupy.ndarray): k vector riêng tương ứng.
#     """
#     n = L.shape[0]
#     V = cp.zeros((n, k + 1), dtype=cp.float32)  # Ma trận chứa các vector trực chuẩn
#     T = cp.zeros((k, k), dtype=cp.float32)  # Ma trận tam trục
    
#     # Khởi tạo v0 ngẫu nhiên và chuẩn hóa
#     v0 = cp.random.randn(n).astype(cp.float32)
#     v0 /= cp.linalg.norm(v0)
#     V[:, 0] = v0
    
#     beta = 0
#     v_prev = cp.zeros(n, dtype=cp.float32)  # Vector trước đó

#     for j in range(min(k, max_iter)):
#         w = L @ V[:, j]  # Nhân ma trận
#         alpha = V[:, j] @ w  # Hệ số chéo chính
#         w -= alpha * V[:, j] + beta * v_prev  # Loại bỏ thành phần trước đó
        
#         beta = cp.linalg.norm(w)  # Hệ số dưới đường chéo
#         if beta < tol:
#             break  # Hội tụ
        
#         V[:, j + 1] = w / beta  # Chuẩn hóa
#         v_prev = V[:, j]  # Lưu lại vector trước
#         T[j, j] = alpha
#         if j < k - 1:
#             T[j, j + 1] = beta
#             T[j + 1, j] = beta

#     # Tính trị riêng của ma trận tam trục T
#     eigvals, eigvecs_T = cp.linalg.eigh(T[:j+1, :j+1])  # Chỉ lấy ma trận kích thước thực sự dùng
    
#     # Chuyển đổi vector riêng từ T_k về không gian gốc
#     eigvecs = V[:, :j+1] @ eigvecs_T[:, :k]
    
#     return eigvals[:k], eigvecs[:, :k]


# def compute_eigen(L, k=2):
#     eigenvalues, eigenvectors = lanczos_method(L, k)
#     return eigenvectors

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
#     D_inv_sqrt = diags(1.0 / cp.sqrt(D_diag))  # Tinh D^-1/2
#     L_normalized = D_inv_sqrt @ L @ D_inv_sqrt  # Chuan hoa ma tran Laplace

#     # Giai bai toan tri rieng bang eigsh
#     eigenvalues, eigenvectors = lanczos_method(L_normalized, k, max_iter=50)

#     # Chuyen lai eigenvectors ve khong gian goc bang cach nhan D^-1/2
#     eigvecs_original = D_inv_sqrt @ eigenvectors

#     return eigvecs_original

def compute_eigen(L, k=2):
    # Tìm các trị riêng nhỏ nhất (Smallest Magnitude)
    eigvals, eigvecs = eigsh(L, k=k, which='SA')  
    return eigvecs





# 4. Gan nhan cho tung diem anh duoc dua tren vector rieng
def assign_labels(eigen_vectors, k):
    # Chuyen du lieu ve CPU de dung K-Means
    eigen_vectors_cpu = eigen_vectors.get()
    logging.info(f"Manh cua vector rieng (9 hang dau):\n{eigen_vectors_cpu[:9, :]}")

    kmeans = KMeans(n_clusters=k, random_state=0).fit(eigen_vectors_cpu)
    labels = kmeans.labels_
    logging.info(f"Nhan gan cho 27 pixel dau tien: {labels[:27]}")
    return cp.array(labels)  # Chuyen lai ve GPU

# 5. Hien thi ket qua
def display_segmentation(image, labels, k):
    h, w, c = image.shape
    segmented_image = cp.zeros_like(cp.array(image), dtype=cp.uint8)
    
    # Tao bang mau ngau nhien
    colors = cp.random.randint(0, 255, size=(k, 3), dtype=cp.uint8)
    
    # To mau tung vung
    for i in range(k):
        segmented_image[labels.reshape(h, w) == i] = colors[i]
    
    # Hien thi tren CPU
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Segmented Image")
    plt.imshow(segmented_image.get())
    plt.axis('off')
    plt.show()

# 6. Ket hop toan bo
def normalized_cuts(image_path, k=2):
    
    # Tinh tong tren GPU
    start_gpu = time.time()
    
    # Doc anh va chuan hoa
    image = io.imread(image_path)
    if image.ndim == 2:  # Neu la anh xam, chuyen thanh RGB
        image = color.gray2rgb(image)
    elif image.shape[2] == 4:  # Neu la anh RGBA, loai bo kenh alpha
        image = image[:, :, :3]
    image = image / 255.0  # Chuan hoa ve [0, 1]
    

    # Tính thời gian riêng cho COO matrix
    start_cpu_coo = time.time()
    # Tinh toan Ncuts
    logging.info("Tinh ma tran trong so...")
    W_sparse = compute_weight_matrix(image)  # Ma trận W ở dạng thưa
    end_cpu_coo = time.time()
    
    logging.info("Tinh Laplace...")
    L, D = compute_laplacian(W_sparse)
    
    logging.info("Tinh eigenvectors...")
    eigen_vectors = compute_eigen(L, k=k)  # Tinh k vector rieng
    
    logging.info("Phan vung do thi...")
    labels = assign_labels(eigen_vectors, k)  # Gan nhan cho moi diem anh
    
    logging.info("Hien thi ket qua...")

    cp.cuda.Stream.null.synchronize()  # Dong bo hoa de dam bao GPU hoan thanh tinh toan
    end_gpu = time.time()
    logging.info(f"Thoi gian: {end_gpu - start_gpu} giay")
    logging.info(f"Thoi gian COO: {end_cpu_coo - start_cpu_coo} giay")

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
