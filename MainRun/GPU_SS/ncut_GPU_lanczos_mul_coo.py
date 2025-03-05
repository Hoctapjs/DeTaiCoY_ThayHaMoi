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
from cupyx.scipy.sparse import coo_matrix


import pandas as pd

def process_logs_for_summary(name, results, run_index):
    """
    Lưu kết quả vào file Excel, đảm bảo mỗi lần chạy có sheet riêng.
    Cập nhật sheet "Tóm tắt" để chỉ chứa trung bình các thời gian của mỗi ảnh.
    """
    excel_file = f"{name}_summary.xlsx"
    
    # Chuyển dữ liệu về DataFrame
    df = pd.DataFrame(results, columns=["Ảnh", "Lần chạy", "Thời gian Lanczos", "Thời gian COO"])
    
    # Mở file để ghi dữ liệu
    with pd.ExcelWriter(excel_file, engine="openpyxl", mode="a" if os.path.exists(excel_file) else "w") as writer:
        sheet_name = f"Sheet {run_index}"  # Tạo sheet theo số lần chạy
        df.to_excel(writer, sheet_name=sheet_name, index=False)

    # Đọc lại toàn bộ dữ liệu từ các sheet để cập nhật "Tóm tắt"
    all_data = []
    with pd.ExcelFile(excel_file) as xls:
        for sheet in xls.sheet_names:
            if sheet.startswith("Sheet "):  # Chỉ lấy dữ liệu từ các lần chạy
                all_data.append(pd.read_excel(xls, sheet_name=sheet))

    if all_data:
        full_df = pd.concat(all_data, ignore_index=True)
        summary = full_df.groupby("Ảnh").agg({
            "Thời gian Lanczos": "mean",
            "Thời gian COO": "mean"
        }).reset_index()

        # 🛠 Sửa lỗi ghi đè bằng `if_sheet_exists="replace"`
        with pd.ExcelWriter(excel_file, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            summary.to_excel(writer, sheet_name="Tóm tắt", index=False)

    print(f"📊 Đã lưu kết quả lần chạy {run_index} vào {excel_file}")

def kiemThuChayNhieuLan(i, name, folder_path):
    if not os.path.isdir(folder_path):
        print(f"❌ Thư mục {folder_path} không tồn tại!")
        return

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"❌ Không tìm thấy file ảnh nào trong {folder_path}!")
        return

    results = []  # Lưu kết quả cho từng lần chạy

    for idx, file_name in enumerate(image_files, start=1):
        image_path = os.path.join(folder_path, file_name)

        print(f"📷 Đang xử lý ảnh {idx}/{len(image_files)}: {image_path}")

        # Gọi hàm xử lý ảnh và lấy thời gian
        total_time, coo_time, lanczos_time = normalized_cuts(image_path)

        # Lưu kết quả
        results.append([file_name, i + 1, lanczos_time, coo_time])

    process_logs_for_summary(name, results, i + 1)  # Truyền số lần chạy vào




# ĐÂY LÀ CÁCH CHẠY MA TRẬN TRỌNG SỐ W TRÊN GPU THEO LOGIC GIỐNG BÊN CPU
def compute_weight_matrix(image, sigma_i=0.1, sigma_x=10):
    h, w, c = image.shape
    coords = cp.array(cp.meshgrid(cp.arange(h), cp.arange(w))).reshape(2, -1).T  # Tọa độ (x, y)
    features = image.reshape(-1, c)  # Đặc trưng màu

    # logging.info(f"Kich thuoc anh: {h}x{w}x{c}")
    # logging.info(f"Kich thuoc dac trung mau: {features.shape}, Kich thuoc toa do: {coords.shape}")
    # logging.info(f"Dac trung mau:\n{features[:9, :9]}")
    # logging.info(f"Toa do:\n{coords[:9, :9]}")

    start_cpu_coo=time.time()
    # Tính độ tương đồng về đặc trưng và không gian
    W_features = rbf_kernel(cp.asnumpy(features), gamma=1/(2 * sigma_i**2))  # Chuyển dữ liệu từ GPU sang CPU
    W_coords = rbf_kernel(cp.asnumpy(coords), gamma=1/(2 * sigma_x**2))  # Chuyển dữ liệu từ GPU sang CPU

    # Chuyển kết quả từ NumPy (CPU) sang CuPy (GPU)
    W_features = cp.asarray(W_features)
    W_coords = cp.asarray(W_coords)
    end_cpu_coo=time.time()

    logging.info(f"Thoi gian COO: {end_cpu_coo - start_cpu_coo} giay")

    W = cp.multiply(W_features, W_coords)  # Phép nhân phần tử của ma trận trên GPU

    # Chuyển thành ma trận thưa dạng COO
    W_sparse = coo_matrix(W)

    # logging.info(f"Kich thuoc ma tran trong so (W): {W.shape}")
    # logging.info(f"Kich thuoc ma tran thua (W_sparse): {W_sparse.shape}")
    # logging.info(f"So luong phan tu khac 0: {W_sparse.nnz}")
    # logging.info(f"Mau cua W_features (9x9 phan tu dau):\n{W_features[:9, :9]}")
    # logging.info(f"Mau cua W_coords (9x9 phan tu dau):\n{W_coords[:9, :9]}")
    # logging.info(f"Mau cua W (9x9 phan tu dau):\n{W[:9, :9]}")

    return W_sparse





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

# CUDA Kernel song song hóa nhân ma trận - vector với 10 luồng
matvec_kernel = cp.RawKernel(r'''
extern "C" __global__
void matvec_mul(const double* A, const double* V, double* W, int N, int chunk_size) {
    int thread_id = threadIdx.x;  
    int start_row = thread_id * chunk_size;  
    int end_row = min(start_row + chunk_size, N); 

    for (int row = start_row; row < end_row; row++) {
        double sum = 0.0;
        for (int col = 0; col < N; col++) {
            sum += A[row * N + col] * V[col];
        }
        W[row] = sum;
    }
}
''', 'matvec_mul')

# # Kernel CUDA để tính tích vô hướng song song
# dot_kernel = cp.RawKernel(r'''
# extern "C" __global__
# void dot_product(const double* A, const double* B, double* result, int N) {
#     __shared__ double cache[1024];  
#     int tid = threadIdx.x + blockIdx.x * blockDim.x;
#     int cacheIndex = threadIdx.x;
    
#     double temp = 0.0;
#     while (tid < N) {
#         temp += A[tid] * B[tid];
#         tid += blockDim.x * gridDim.x;
#     }
    
#     cache[cacheIndex] = temp;
#     __syncthreads();

#     // Reduction trong shared memory
#     for (int s = blockDim.x / 2; s > 0; s >>= 1) {
#         if (cacheIndex < s) {
#             cache[cacheIndex] += cache[cacheIndex + s];
#         }
#         __syncthreads();
#     }
    
#     if (cacheIndex == 0) {
#         atomicAdd(result, cache[0]);
#     }
# }
# ''', 'dot_product')


def Lanczos(A, v, m, num_threads=256):
    """
    Thuật toán Lanczos song song với CuPy và giới hạn số luồng.
    :param A: Ma trận vuông (numpy 2D array hoặc CuPy array).
    :param v: Vector khởi tạo.
    :param m: Số bước Lanczos.
    :param num_threads: Số luồng CUDA sẽ sử dụng.
    :return: Ma trận tam giác T và ma trận trực giao V.
    """
    n = len(v)
    V = cp.zeros((m, n), dtype=cp.float64)
    T = cp.zeros((m, m), dtype=cp.float64)

    # Chuyển dữ liệu sang GPU
    A = cp.asarray(A, dtype=cp.float64)
    v = cp.asarray(v, dtype=cp.float64)

    # Chuẩn hóa vector đầu vào
    V[0, :] = v / cp.linalg.norm(v)
    
    # Khởi tạo vector w (kết quả nhân ma trận - vector)
    w = cp.zeros(n, dtype=cp.float64)

    # Kích thước mỗi luồng xử lý
    chunk_size = (n + num_threads - 1) // num_threads  # Chia đều công việc

    # Gọi kernel với 1 block và num_threads thread
    matvec_kernel((1,), (num_threads,), (A, V[0, :], w, n, chunk_size))
    
    alpha = cp.dot(w, V[0, :])
    w = w - alpha * V[0, :]
    T[0, 0] = alpha  # Gán giá trị alpha vào ma trận T

    for j in range(1, m):
        beta = cp.linalg.norm(w)
        if beta < 1e-10:
            break
        V[j, :] = w / beta

        # Tính w = A @ V[j, :] bằng CUDA với 10 luồng
        matvec_kernel((1,), (num_threads,), (A, V[j, :], w, n, chunk_size))
        
        alpha = cp.dot(w, V[j, :])
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
    D_inv_sqrt = diags(1.0 / cp.sqrt(D_diag))  # Tinh D^-1/2
    L_normalized = D_inv_sqrt @ L @ D_inv_sqrt  # Chuan hoa ma tran Laplace
    
    # Khởi tạo vector ngẫu nhiên
    v0 = cp.random.rand(L.shape[0])
    v0 /= cp.linalg.norm(v0)
     
    start_lan = time.time()
    # Áp dụng thuật toán Lanczos
    T, V = Lanczos(L_normalized, v0, m=k+5)  # Sử dụng m > k để tăng độ chính xác
    end_lan = time.time()
    # logging.info(f"Thoi gian lanczos: {end_lan - start_lan} giay")
    lanczos_time = end_lan-start_lan;
    # Tính trị riêng và vector riêng của ma trận tam giác T
    eigvals, eigvecs_T = cp.linalg.eigh(T[:k, :k])
    
    # Chuyển đổi vector riêng về không gian gốc
    eigvecs_original = D_inv_sqrt @ (V[:k, :].T @ eigvecs_T)
    
    return eigvecs_original, lanczos_time

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
    start_gpu = time.time()

    # Đọc ảnh
    image = io.imread(image_path)
    if image.ndim == 2:
        image = color.gray2rgb(image)
    elif image.shape[2] == 4:
        image = image[:, :, :3]
    image = image / 255.0

    start_coo = time.time()
    W = compute_weight_matrix(image)
    end_coo = time.time()

    L, D = compute_laplacian(W)
    eigen_vectors,lanczos_time  = compute_eigen(L, D, k=k)
    labels = assign_labels(eigen_vectors, k)

    cp.cuda.Stream.null.synchronize()
    end_gpu = time.time()

    total_time = end_gpu - start_gpu
    coo_time = end_coo - start_coo

    return total_time, coo_time, lanczos_time


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
