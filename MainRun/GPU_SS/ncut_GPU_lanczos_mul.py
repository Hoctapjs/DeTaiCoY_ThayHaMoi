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
import os
import re


def process_logs_for_summary(name):
    """
    Xử lý log file để tính thời gian trung bình của từng ảnh và toàn bộ thư mục.
    """
    time_per_image = {}
    coo_time_per_image = {}

    log_files = [f for f in os.listdir() if f.startswith(name) and f.endswith(".txt") and "summary" not in f]

    for log_file in log_files:
        with open(log_file, "r", encoding="utf-8") as f:
            last_coo_time = None
            last_time = None
            last_image = None

            for line in f:
                match_coo = re.search(r"Thoi gian COO: ([\d.]+) giay", line)
                if match_coo:
                    last_coo_time = float(match_coo.group(1))

                match_time = re.search(r"Thoi gian: ([\d.]+) giay", line)
                if match_time:
                    last_time = float(match_time.group(1))

                match_image = re.search(r"Anh (\d+\.jpg) lan \d+", line)
                if match_image:
                    last_image = match_image.group(1)

                if last_image and last_coo_time is not None and last_time is not None:
                    if last_image not in time_per_image:
                        time_per_image[last_image] = {"total_time": 0, "count": 0}
                        coo_time_per_image[last_image] = {"total_coo_time": 0, "count": 0}

                    time_per_image[last_image]["total_time"] += last_time
                    time_per_image[last_image]["count"] += 1

                    coo_time_per_image[last_image]["total_coo_time"] += last_coo_time
                    coo_time_per_image[last_image]["count"] += 1

                    last_coo_time = None
                    last_time = None
                    last_image = None

    avg_time_per_image = {
        img: time["total_time"] / time["count"]
        for img, time in time_per_image.items()
    }

    avg_coo_time_per_image = {
        img: time["total_coo_time"] / time["count"]
        for img, time in coo_time_per_image.items()
    }

    avg_time_folder = sum(avg_time_per_image.values()) / len(avg_time_per_image) if avg_time_per_image else 0
    avg_coo_time_folder = sum(avg_coo_time_per_image.values()) / len(avg_coo_time_per_image) if avg_coo_time_per_image else 0

    summary_log = f"{name}_summary.txt"
    with open(summary_log, "w", encoding="utf-8") as f:
        f.write("Thời gian trung bình của từng ảnh:\n\n")
        for img in avg_time_per_image:
            f.write(f"{img}: {avg_time_per_image[img]:.4f} giây (Lanczos) | {avg_coo_time_per_image[img]:.4f} giây (COO)\n")

        f.write(f"\nThời gian trung bình của cả thư mục:\n")
        f.write(f"Lanczos: {avg_time_folder:.4f} giây\n")
        f.write(f"COO: {avg_coo_time_folder:.4f} giây\n")

    print(f"📊 Đã lưu kết quả vào {summary_log}")



def kiemThuChayNhieuLan(i, name, folder_path):
    if not os.path.isdir(folder_path):
        print(f"❌ Thư mục {folder_path} không tồn tại!")
        return

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"❌ Không tìm thấy file ảnh nào trong {folder_path}!")
        return

    # Danh sách lưu thời gian của mỗi ảnh
    time_per_image = {img: [] for img in image_files}

    for idx, file_name in enumerate(image_files, start=1):
        image_path = os.path.join(folder_path, file_name)

        log_file = f"{name}_{i}_{idx}.txt"

        # Cấu hình logging
        logging.basicConfig(filename=log_file, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

        print(f"📷 Đang xử lý ảnh {idx}/{len(image_files)}: {image_path}")

        # Gọi hàm xử lý ảnh
        normalized_cuts(image_path)

        # Ghi vào log
        logging.info(f"Anh {file_name} lan {i+1}")

    process_logs_for_summary(name)
# ĐÂY LÀ CÁCH CHẠY MA TRẬN TRỌNG SỐ W TRÊN GPU THEO LOGIC GIỐNG BÊN CPU
def compute_weight_matrix(image, sigma_i=0.1, sigma_x=10):
    h, w, c = image.shape
    # logging.info(f"Kích thước ảnh: {h}x{w}x{c}")

    # Tọa độ (x, y)
    coords = cp.array(cp.meshgrid(cp.arange(h), cp.arange(w))).reshape(2, -1).T

    # Đặc trưng màu
    features = cp.array(image).reshape(-1, c)

    # logging.info(f"Kích thước đặc trưng màu: {features.shape}, Kích thước tọa độ: {coords.shape}")
    # logging.info(f"Đặc trưng màu (9 phần tử đầu):\n{features[:9, :9]}")
    # logging.info(f"Tọa độ (9 phần tử đầu):\n{coords[:9, :9]}")

    # Tính ma trận trọng số bằng vector hóa
    W_color = cp.array(rbf_kernel(features.get(), gamma=1 / (2 * sigma_i**2)))
    W_space = cp.array(rbf_kernel(coords.get(), gamma=1 / (2 * sigma_x**2)))
    W = W_color * W_space

    # logging.info(f"Mảnh của W (9x9 phần tử đầu):\n{W[:9, :9]}")
    return W


# 2. Tinh ma tran Laplace
def compute_laplacian(W):
    D = cp.diag(W.sum(axis=1))  # Ma trận đường chéo
    L = D - W
    # logging.info("Kích thước ma trận đường chéo (D):", D.shape)
    # logging.info("Mẫu của D (9x9 phần tử đầu):\n", D[:9, :9])
    # logging.info("Kích thước ma trận Laplace (L):", L.shape)
    # logging.info("Mẫu của L (9x9 phần tử đầu):\n", L[:9, :9])
    
    return L, D

# 3. Giai bai toan tri rieng

# Kernel CUDA để nhân ma trận với vector
matvec_kernel = cp.RawKernel(r'''
extern "C" __global__
void matvec_mul(const double* A, const double* V, double* W, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        double sum = 0.0;
        for (int col = 0; col < N; col++) {
            sum += A[row * N + col] * V[col];
        }
        W[row] = sum;
    }
}
''', 'matvec_mul')

# Kernel CUDA để tính tích vô hướng song song
dot_kernel = cp.RawKernel(r'''
extern "C" __global__
void dot_product(const double* A, const double* B, double* result, int N) {
    __shared__ double cache[1024];  
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    
    double temp = 0.0;
    while (tid < N) {
        temp += A[tid] * B[tid];
        tid += blockDim.x * gridDim.x;
    }
    
    cache[cacheIndex] = temp;
    __syncthreads();

    // Reduction trong shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (cacheIndex < s) {
            cache[cacheIndex] += cache[cacheIndex + s];
        }
        __syncthreads();
    }
    
    if (cacheIndex == 0) {
        atomicAdd(result, cache[0]);
    }
}
''', 'dot_product')


def Lanczos(A, v, m):
    n = len(v)
    V = cp.zeros((m, n), dtype=cp.float64)
    T = cp.zeros((m, m), dtype=cp.float64)
    V[0, :] = v / cp.linalg.norm(v)

    W = cp.zeros(n, dtype=cp.float64)  # Vector tạm thời
    alpha_gpu = cp.zeros(1, dtype=cp.float64)  # Giá trị alpha trên GPU

    for j in range(m-1):
        # Nhân ma trận A với vector V[j, :] song song trên CUDA
        matvec_kernel((n // 256 + 1,), (256,), (A, V[j, :], W, n))

        # Tính alpha = dot(W, V[j, :]) song song trên CUDA
        alpha_gpu.fill(0)
        dot_kernel((n // 256 + 1,), (256,), (W, V[j, :], alpha_gpu, n))
        alpha = alpha_gpu  # Giữ trên GPU

        W -= alpha * V[j, :]

        if j > 0:
            W -= beta * V[j - 1, :]

        beta = cp.linalg.norm(W)
        if beta < 1e-10:
            break

        if j + 1 < V.shape[0]:
            V[j + 1, :] = W / beta
        else:
            print(f"Lỗi: j + 1 ({j + 1}) vượt quá giới hạn {V.shape[0]}")

        T[j, j] = alpha
        if j < m - 1:
            T[j, j + 1] = beta
            T[j + 1, j] = beta

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
    D_inv_sqrt = diags(1.0 / cp.sqrt(D_diag))  # Tinh D^-1/2
    L_normalized = D_inv_sqrt @ L @ D_inv_sqrt  # Chuan hoa ma tran Laplace
    
    # Khởi tạo vector ngẫu nhiên
    v0 = cp.random.rand(L.shape[0])
    v0 /= cp.linalg.norm(v0)
    
    start_lan = time.time()
    # Áp dụng thuật toán Lanczos
    T, V = Lanczos(L_normalized, v0, m=k+5)  # Sử dụng m > k để tăng độ chính xác
    end_lan = time.time()
    logging.info(f"Thoi gian: {end_lan - start_lan} giay")
    
    # Tính trị riêng và vector riêng của ma trận tam giác T
    eigvals, eigvecs_T = cp.linalg.eigh(T[:k, :k])
    
    # Chuyển đổi vector riêng về không gian gốc
    eigvecs_original = D_inv_sqrt @ (V[:k, :].T @ eigvecs_T)
    
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
    
    # Tinh toan Ncuts
    start_cpu_coo = time.time()
    # logging.info("Dang tinh toan ma tran trong so...")
    W = compute_weight_matrix(image)
    end_cpu_coo = time.time()
    
    # logging.info("Tinh Laplace...")
    L, D = compute_laplacian(W)
    
    # logging.info("Tinh eigenvectors...")
    eigen_vectors = compute_eigen(L,D, k=k)  # Tinh k vector rieng
    
    # logging.info("Phan vung do thi...")
    labels = assign_labels(eigen_vectors, k)  # Gan nhan cho moi diem anh
    
    # logging.info("Hien thi ket qua...")

    cp.cuda.Stream.null.synchronize()  # Dong bo hoa de dam bao GPU hoan thanh tinh toan
    end_gpu = time.time()
    logging.info(f"Thoi gian COO: {end_cpu_coo - start_cpu_coo} giay")
    # logging.info(f"Thoi gian: {end_gpu - start_gpu} giay")

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
