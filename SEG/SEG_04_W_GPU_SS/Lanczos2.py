import cupy as cp
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import KMeans
from skimage import io, color
import cupyx.scipy.sparse as sp
from cupyx.scipy.sparse import diags
import time
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import logging
import os
import re
import pandas as pd
from datetime import datetime



# Hàm lưu file .seg (giữ nguyên)
def save_seg_file(labels, image_shape, output_path, image_name="image"):
    h, w = image_shape[:2]
    labels_2d = labels.reshape(h, w)
    unique_labels = cp.unique(labels)
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

# Hàm xử lý và lưu kết quả Excel
def process_logs_for_summary(name, results, run_index):
    excel_file = f"{name}_summary.xlsx"
    df = pd.DataFrame(results, columns=["Ảnh", "Lần chạy", "Thời gian Lanczos", "Thời gian COO"])
    
    with pd.ExcelWriter(excel_file, engine="openpyxl", mode="a" if os.path.exists(excel_file) else "w") as writer:
        sheet_name = f"Sheet {run_index}"
        df.to_excel(writer, sheet_name=sheet_name, index=False)

    all_data = []
    with pd.ExcelFile(excel_file) as xls:
        for sheet in xls.sheet_names:
            if sheet.startswith("Sheet "):
                all_data.append(pd.read_excel(xls, sheet_name=sheet))

    if all_data:
        full_df = pd.concat(all_data, ignore_index=True)
        summary = full_df.groupby("Ảnh").agg({
            "Thời gian Lanczos": "mean",
            "Thời gian COO": "mean"
        }).reset_index()

        with pd.ExcelWriter(excel_file, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            summary.to_excel(writer, sheet_name="Tóm tắt", index=False)

    print(f"📊 Đã lưu kết quả lần chạy {run_index} vào {excel_file}")

# Hàm kiểm tra chạy nhiều lần
def kiemThuChayNhieuLan(i, name, folder_path):
    if not os.path.isdir(folder_path):
        print(f"❌ Thư mục {folder_path} không tồn tại!")
        return

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"❌ Không tìm thấy file ảnh nào trong {folder_path}!")
        return

    results = []
    for idx, file_name in enumerate(image_files, start=1):
        image_path = os.path.join(folder_path, file_name)
        print(f"📷 Đang xử lý ảnh {idx}/{len(image_files)}: {image_path}")
        total_time, coo_time, lanczos_time = normalized_cuts(i, image_path)
        results.append([file_name, i + 1, lanczos_time, coo_time])

    process_logs_for_summary(name, results, i + 1)

# Tính ma trận trọng số
def compute_weight_matrix(image, sigma_i=0.2, sigma_x=15.0):
    h, w, c = image.shape
    coords = cp.array(cp.meshgrid(cp.arange(h), cp.arange(w))).reshape(2, -1).T
    features = cp.array(image).reshape(-1, c)

    W_color = cp.array(rbf_kernel(features.get(), gamma=1 / (2 * sigma_i**2)))
    W_space = cp.array(rbf_kernel(coords.get(), gamma=1 / (2 * sigma_x**2)))
    W = W_color * W_space
    return W

# Tính ma trận Laplacian
def compute_laplacian(W):
    D = cp.diag(W.sum(axis=1))
    L = D - W
    return L, D

# Hàm tích vô hướng
def handle_dot(a, b):
    return cp.dot(a, b)



# Định nghĩa CUDA kernel song song hóa nhân ma trận-vector với 10 luồng
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


# Hàm nhân ma trận-vector sử dụng CUDA kernel
def matrix_vector_product(A, v):
    N = A.shape[0]
    W = cp.zeros(N, dtype=cp.float64)  # Vector kết quả
    
    # Cấu hình kernel: 10 luồng, không dùng block
    threads_per_block = 10
    chunk_size = (N + threads_per_block - 1) // threads_per_block  # Chia đều công việc cho 10 luồng
    
    # Đảm bảo A là mảng liền kề (contiguous) để kernel hoạt động đúng
    A_contiguous = cp.ascontiguousarray(A)
    
    # Gọi kernel với 1 block và 10 threads
    matvec_kernel((1,), (threads_per_block,), (A_contiguous, v, W, N, chunk_size))
    
    return W

# Thuật toán Lanczos với trực giao hóa chặt chẽ
def Lanczos(A, v, m):
    n = len(v)
    V = cp.zeros((m, n), dtype=cp.float64)
    T = cp.zeros((m, m), dtype=cp.float64)
    V[0, :] = v / cp.linalg.norm(v)
    
    w = matrix_vector_product(A, V[0, :])
    alpha = handle_dot(w, V[0, :])
    w = w - alpha * V[0, :]
    T[0, 0] = alpha
    
    for j in range(1, m):
        beta = cp.linalg.norm(w)
        if beta < 1e-10:
            break
        V[j, :] = w / beta
        # Trực giao hóa (Gram-Schmidt) chặt chẽ
        for i in range(j):
            proj = handle_dot(V[j, :], V[i, :])
            V[j, :] -= proj * V[i, :]
        norm_vj = cp.linalg.norm(V[j, :])
        if norm_vj < 1e-10:
            break
        V[j, :] /= norm_vj
        
        w = matrix_vector_product(A, V[j, :])
        alpha = handle_dot(w, V[j, :])
        w = w - alpha * V[j, :] - (beta * V[j-1, :] if j > 0 else 0)
        
        T[j, j] = alpha
        if j > 0:
            T[j-1, j] = beta
            T[j, j-1] = beta
    
    # Đảm bảo T đối xứng
    T = (T + T.T) / 2
    return T[:j+1, :j+1], V[:j+1, :]

# Tính trị riêng và vector riêng
def compute_eigen(L, D, k=2):
    D_diag = D.diagonal().copy()
    D_diag[D_diag < 1e-10] = 1e-10
    D_inv_sqrt = diags(1.0 / cp.sqrt(D_diag))
    L_normalized = D_inv_sqrt @ L @ D_inv_sqrt
    
    v0 = cp.random.rand(L.shape[0])
    v0 /= cp.linalg.norm(v0)
    
    start_lan = time.time()
    m = k + 50  # Tăng số lần lặp để cải thiện độ chính xác
    T, V = Lanczos(L_normalized, v0, m)
    end_lan = time.time()
    lanczos_time = end_lan - start_lan
    
    # Tính trị riêng và vector riêng của ma trận T
    eigvals, eigvecs_T = cp.linalg.eigh(T)
    idx = cp.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs_T = eigvecs_T[:, idx]
    
    # Bỏ vector riêng nhỏ nhất (trị riêng gần 0)
    eigvecs_T = eigvecs_T[:, 1:k+1]
    eigvecs_original = D_inv_sqrt @ (V.T @ eigvecs_T)
    
    # Chuẩn hóa vector riêng
    for i in range(eigvecs_original.shape[1]):
        norm = cp.linalg.norm(eigvecs_original[:, i])
        if norm > 1e-10:
            eigvecs_original[:, i] /= norm
    
    logging.info(f"Thời gian Lanczos: {lanczos_time:.6f} giây")
    logging.info(f"Trị riêng: {eigvals[:k+1].get()}")
    logging.info(f"Vector riêng (mẫu): {eigvecs_original[:5, :].get()}")
    return eigvecs_original, lanczos_time

# Gán nhãn
def assign_labels(eigen_vectors, k):
    eigen_vectors_cpu = eigen_vectors.get()
    kmeans = KMeans(n_clusters=k, random_state=0).fit(eigen_vectors_cpu)
    labels = kmeans.labels_
    return cp.array(labels)

# Hiển thị phân đoạn (giữ nguyên)
def display_segmentation(image, labels, k):
    h, w, c = image.shape
    segmented_image = cp.zeros_like(cp.array(image), dtype=cp.uint8)
    colors = cp.random.randint(0, 255, size=(k, 3), dtype=cp.uint8)
    
    for i in range(k):
        segmented_image[labels.reshape(h, w) == i] = colors[i]
    
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

# Hàm chính xử lý ảnh
def normalized_cuts(lan, image_path, k=2):
    start_gpu = time.time()

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
    eigen_vectors, lanczos_time = compute_eigen(L, D, k=k)
    labels = assign_labels(eigen_vectors, k)

    # Lưu file .seg
    imagename = os.path.splitext(os.path.basename(image_path))[0]
    seg_output_path = f"{imagename}_segmentation_{lan}.seg"
    save_seg_file(labels.get(), image.shape, seg_output_path, imagename)

    cp.cuda.Stream.null.synchronize()
    end_gpu = time.time()

    total_time = end_gpu - start_gpu
    coo_time = end_coo - start_coo

    return total_time, coo_time, lanczos_time

# Hàm mở dialog chọn file
def open_file_dialog():
    root = Tk()
    root.withdraw()
    file_path = askopenfilename(title="Chọn ảnh", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
    return file_path

if __name__ == "__main__":
    image_path = open_file_dialog()
    if image_path:
        logging.info(f"Đã chọn ảnh: {image_path}")
        normalized_cuts(1, image_path, k=3)  # Thêm lan=1
    else:
        logging.info("Không có ảnh nào được chọn.")