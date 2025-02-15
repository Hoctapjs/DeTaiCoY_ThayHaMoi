import cupy as cp  # Thay thế NumPy bằng CuPy
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
# from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from skimage import io, color
import cupyx.scipy.sparse as sp
from cupyx.scipy.sparse.linalg import eigsh
import time
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import logging
import os
import re
from  cupyx.scipy.sparse import diags
from cupyx.scipy.sparse import coo_matrix

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





# CUDA Kernel để tính RBF Kernel song song
rbf_kernel_cuda = cp.RawKernel(r'''
extern "C" __global__
void rbf_kernel(const double* X, double* W, int n, int d, double gamma) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    for (int j = 0; j < n; j++) {
        double dist = 0.0;
        for (int k = 0; k < d; k++) {
            double diff = X[i * d + k] - X[j * d + k];
            dist += diff * diff;
        }
        W[i * n + j] = exp(-gamma * dist);
    }
}
''', 'rbf_kernel')

def compute_rbf_matrix(X, gamma, threads_per_block=1024):
    n, d = X.shape
    X_gpu = cp.asarray(X, dtype=cp.float64)
    W_gpu = cp.zeros((n, n), dtype=cp.float64)

    # Tính số block cần thiết
    num_blocks = (n + threads_per_block - 1) // threads_per_block

    # Gọi CUDA Kernel
    rbf_kernel_cuda((num_blocks,), (threads_per_block,), (X_gpu, W_gpu, n, d, gamma))

    return W_gpu


# 1. Tinh ma tran trong so
def compute_weight_matrix(image, sigma_i=0.1, sigma_x=10, threads_per_block=300):
    h, w, c = image.shape
    coords = cp.array(cp.meshgrid(cp.arange(h), cp.arange(w))).reshape(2, -1).T  # Tọa độ (x, y)
    features = cp.array(image.reshape(-1, c))  # Chuyển toàn bộ đặc trưng lên GPU

    # logging.info(f"Kích thước ảnh: {h}x{w}x{c}")
    # logging.info(f"Kích thước đặc trưng màu: {features.shape}, Kích thước tọa độ: {coords.shape}")

    gamma_i = 1 / (2 * sigma_i**2)
    gamma_x = 1 / (2 * sigma_x**2)

    W_features = compute_rbf_matrix(features, gamma_i, threads_per_block)
    W_coords = compute_rbf_matrix(coords, gamma_x, threads_per_block)

    W = cp.multiply(W_features, W_coords)
    return W



# 2. Tinh ma tran Laplace
def compute_laplacian(W_sparse):
    # Tổng của các hàng trong ma trận W
    D_diag = W_sparse.sum(axis=1).get().flatten()  # Tính tổng các hàng
    D = cp.diag(D_diag)  # Tạo ma trận đường chéo từ tổng hàng
    L = D - W_sparse  # L = D - W
    # logging.info("Kich thuoc ma tran duong cheo (D): %s", D.shape)
    # logging.info("Mau cua D (9 phan tu dau):\n%s", D_diag[:9])  # In phần tử trên đường chéo
    # logging.info("Kich thuoc ma tran Laplace (L): %s", L.shape)
    # logging.info("Mau cua L (9x9 phan tu dau):\n%s", L[:9, :9])
    return L, D

# 3. Giai bai toan tri rieng
def compute_eigen(L, D, k=2):
    """
    Giai bai toan tri rieng bang thuat toan Lanczos (eigsh) tren GPU.
    :param L: Ma tran Laplace thua (CuPy sparse matrix).
    :param D: Ma tran duong cheo (CuPy sparse matrix).
    :param k: So tri rieng nho nhat can tinh.
    :return: Cac vector rieng tuong ung (k vector).
    """
    # Chuan hoa ma tran Laplace: D^-1/2 * L * D^-1/2
    D_diag = D.diagonal().copy()  # Lay duong cheo cua D
    D_diag[D_diag < 1e-10] = 1e-10  # Tranh chia cho 0 hoac gan 0
    D_inv_sqrt = diags(1.0 / cp.sqrt(D_diag))  # Tinh D^-1/2
    L_normalized = D_inv_sqrt @ L @ D_inv_sqrt  # Chuan hoa ma tran Laplace

    # Giai bai toan tri rieng bang eigsh
    eigvals, eigvecs = eigsh(L_normalized, k=k, which='SA')  # Dung SA thay vi SM

    # Chuyen lai eigenvectors ve khong gian goc bang cach nhan D^-1/2
    eigvecs_original = D_inv_sqrt @ eigvecs

    return eigvecs_original

# 4. Gan nhan cho tung diem anh duoc dua tren vector rieng
def assign_labels(eigen_vectors, k):
    # Chuyen du lieu ve CPU de dung K-Means
    eigen_vectors_cpu = eigen_vectors.get()
    # logging.info(f"Manh cua vector rieng (9 hang dau):\n{eigen_vectors_cpu[:9, :]}")

    kmeans = KMeans(n_clusters=k, random_state=0).fit(eigen_vectors_cpu)
    labels = kmeans.labels_
    # logging.info(f"Nhan gan cho 27 pixel dau tien: {labels[:27]}")
    return cp.array(labels)  # Chuyen lai ve GPU

def save_segmentation(image, labels, k, output_path):
    h, w, c = image.shape
    segmented_image = cp.zeros_like(image, dtype=cp.uint8)
    for i in range(k):
        mask = labels.reshape(h, w) == i
        cluster_pixels = image.get()[mask]
        mean_color = (cluster_pixels.mean(axis=0) * 255).astype(cp.uint8) if len(cluster_pixels) > 0 else cp.array([0, 0, 0], dtype=cp.uint8)
        segmented_image[mask] = mean_color
    io.imsave(output_path, segmented_image)

# 6. Ket hop toan bo
def normalized_cuts(lan, imagename, image_path, output_path):
    
    # Tinh toan tren GPU
    start_gpu = time.time()
    logging.info(f"file name: {imagename}")
    logging.info(f"Lan thu: {lan}")
    # Doc anh va chuan hoa
    image = io.imread(image_path)
    if image.ndim == 2:  # Neu la anh xam, chuyen thanh RGB
        image = color.gray2rgb(image)
    elif image.shape[2] == 4:  # Neu la anh RGBA, loai bo kenh alpha
        image = image[:, :, :3]
    image = image / 255.0  # Chuan hoa ve [0, 1]
    
    k=3

    # Tinh toan Ncuts
    start_cpu_coo = time.time()
    W = compute_weight_matrix(image)
    end_cpu_coo = time.time()

    L, D = compute_laplacian(W)
    
    eigen_vectors = compute_eigen(L,D, k=k)  # Tinh k vector rieng
    
    labels = assign_labels(eigen_vectors, k)  # Gan nhan cho moi diem anh
    # save_segmentation(image, labels, k, output_path)

    cp.cuda.Stream.null.synchronize()  # Dong bo hoa de dam bao GPU hoan thanh tinh toan
    end_gpu = time.time()
    logging.info(f"Thoi gian: {end_gpu - start_gpu} giay")
    logging.info(f"Thoi gian W: {end_cpu_coo - start_cpu_coo} giay")
    
    # ✅ Giải phóng bộ nhớ GPU sau khi sử dụng
    del W, L, D, eigen_vectors
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    cp.cuda.Device(0).synchronize()  # Đảm bảo giải phóng hoàn toàn

    # display_segmentation(image, labels, k)


# # 8. Chay thu nghiem
# if __name__ == "__main__":
