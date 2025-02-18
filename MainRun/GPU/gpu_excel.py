import cupy as cp  # Thay thế NumPy bằng CuPy
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from cupyx.scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from skimage import io, color
import os
import time
import pandas as pd

# Hàm xử lý nhiều lần chạy
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
        time_taken, time_w = normalized_cuts(i, file_name, image_path)

        # Lưu kết quả vào danh sách
        results.append([i, idx, file_name, time_taken, time_w])

    # Ghi kết quả vào file Excel
    df = pd.DataFrame(results, columns=["Lần chạy", "Ảnh số", "Tên ảnh", "Thời gian tổng (s)", "Thời gian tính W (s)"])
    log_file = f"{name}_{i}_{idx}.txt"

    output_excel= "result"
    output_excel= f"{output_excel}_{i}_{idx}.xlsx"
    
    df.to_excel(output_excel, index=False, engine='openpyxl')
    print(f"✅ Kết quả đã lưu vào {output_excel}")

# Tính ma trận trọng số W
def compute_weight_matrix(image, sigma_i=0.1, sigma_x=10):
    h, w, c = image.shape

    coords = cp.array(cp.meshgrid(cp.arange(h), cp.arange(w))).reshape(2, -1).T
    features = cp.array(image).reshape(-1, c)

    start_w = time.time()
    W_color = cp.array(rbf_kernel(features.get(), gamma=1 / (2 * sigma_i**2)))
    W_space = cp.array(rbf_kernel(coords.get(), gamma=1 / (2 * sigma_x**2)))
    W = W_color * W_space
    end_w = time.time()

    return W, end_w - start_w  # Trả về cả ma trận W và thời gian tính

# Tính ma trận Laplace
def compute_laplacian(W):
    D = cp.diag(W.sum(axis=1))
    L = D - W
    return L, D

# Giải bài toán trị riêng
def compute_eigen(L, k=2):
    eigvals, eigvecs = eigsh(L, k=k, which='SA')  
    return eigvecs

# Gán nhãn bằng K-Means
def assign_labels(eigen_vectors, k):
    eigen_vectors_cpu = eigen_vectors.get()
    kmeans = KMeans(n_clusters=k, random_state=0).fit(eigen_vectors_cpu)
    labels = kmeans.labels_
    return cp.array(labels)  # Chuyển lại về GPU

# Xử lý ảnh bằng thuật toán Normalized Cuts
def normalized_cuts(lan, imagename, image_path):
    start_gpu = time.time()
    
    # Đọc ảnh
    image = io.imread(image_path)
    if image.ndim == 2:  
        image = color.gray2rgb(image)
    elif image.shape[2] == 4:  
        image = image[:, :, :3]
    image = image / 255.0  

    k = 3
    W, time_w = compute_weight_matrix(image)
    L, D = compute_laplacian(W)
    
    eigen_vectors = compute_eigen(L, k=k)
    labels = assign_labels(eigen_vectors, k)

    cp.cuda.Stream.null.synchronize()  
    end_gpu = time.time()

    # Giải phóng bộ nhớ GPU
    del W, L, D, eigen_vectors
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    cp.cuda.Device(0).synchronize()

    return end_gpu - start_gpu, time_w  # Trả về thời gian tổng và thời gian tính W
