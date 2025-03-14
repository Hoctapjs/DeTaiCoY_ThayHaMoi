import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse.linalg import eigsh
from scipy.sparse import diags
from sklearn.cluster import KMeans
from skimage import io, color
import time
import os
import pandas as pd

def kiemThuChayNhieuLan(i, name, folder_path, output_excel="results.xlsx"):
    if not os.path.isdir(folder_path):
        print(f"❌ Thư mục {folder_path} không tồn tại!")
        return
    
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    if not image_files:
        print(f"❌ Không tìm thấy file ảnh nào trong {folder_path}!")
        return

    results = []
    for idx, file_name in enumerate(image_files, start=1):
        image_path = os.path.join(folder_path, file_name)
        print(f"📷 Đang xử lý ảnh {idx}: {image_path}")
        
        total_time, wf_time, wc_time, W_all = normalized_cuts(i, file_name, image_path)
        results.append([i, idx, file_name, wf_time, wc_time, W_all])

    df = pd.DataFrame(results, columns=["Lần chạy", "Ảnh số", "Tên ảnh", "Thời gian W đặc trưng (s)", "Thời gian W tọa độ (s)", "Thời gian W All"])
    output_excel = f"result_{name}_{i}.xlsx"
    df.to_excel(output_excel, index=False, engine='openpyxl')
    print(f"✅ Kết quả đã lưu vào {output_excel}")

def compute_weight_matrix(image, sigma_i=0.1, sigma_x=10):
    h, w, c = image.shape
    coords = np.array(np.meshgrid(np.arange(h), np.arange(w))).reshape(2, -1).T  # Tọa độ (x, y)
    features = image.reshape(-1, c)  # Chuyển toàn bộ đặc trưng thành mảng 2D
    gamma_i = 1 / (2 * sigma_i**2)
    gamma_x = 1 / (2 * sigma_x**2)
    
    start_features = time.time()
    W_features = rbf_kernel(features, gamma=gamma_i)
    end_features = time.time()
    W_features_time = end_features - start_features
    
    start_coords = time.time()
    W_coords = rbf_kernel(coords, gamma=gamma_x)
    end_coords = time.time()
    W_coords_time = end_coords - start_coords
    
    W = np.multiply(W_features, W_coords)
    return W, W_features_time, W_coords_time

def compute_laplacian(W):
    D = np.diag(W.sum(axis=1))
    L = D - W
    return L, D

from scipy.sparse import diags, issparse
import numpy as np
import time
from scipy.sparse.linalg import eigsh

def compute_eigen(L, D, k=2):
    if issparse(D):
        D = D.todense()  # Chuyển sang dạng dense nếu là sparse

    D_diag = np.array(D.diagonal()).copy()  # Bản sao có thể chỉnh sửa
    D_diag[D_diag < 1e-10] = 1e-10  # Tránh giá trị quá nhỏ gây lỗi

    if issparse(D):
        D.setdiag(D_diag)  # Cập nhật đường chéo nếu là sparse matrix
    else:
        np.fill_diagonal(D, D_diag)  # Nếu là numpy array, dùng fill_diagonal()

    D_inv_sqrt = diags(1.0 / np.sqrt(D_diag))  # Tạo ma trận nghịch đảo căn
    L_normalized = D_inv_sqrt @ L @ D_inv_sqrt  # Chuẩn hóa Laplacian

    start_time = time.time()
    eigvals, eigvecs = eigsh(L_normalized, k, which='SM')  # Tính eigen
    end_time = time.time()

    lanczos_time = end_time - start_time
    eigvecs_original = D_inv_sqrt @ eigvecs  # Khôi phục eigen vectors gốc

    return eigvecs_original, lanczos_time


def assign_labels(eigen_vectors, k):
    return KMeans(n_clusters=k, random_state=0).fit(eigen_vectors).labels_

# tạo seg

def save_segmentation(image, labels, k, output_path):
    h, w, c = image.shape
    segmented_image = np.zeros_like(image, dtype=np.uint8)
    for i in range(k):
        mask = labels.reshape(h, w) == i
        cluster_pixels = image[mask]
        mean_color = (cluster_pixels.mean(axis=0) * 255).astype(np.uint8) if len(cluster_pixels) > 0 else np.array([0, 0, 0], dtype=np.uint8)
        segmented_image[mask] = mean_color
    io.imsave(output_path, segmented_image)

from datetime import datetime

def save_seg_file(labels, image_shape, output_path, image_name="image"):
    h, w = image_shape[:2]
    unique_labels = np.unique(labels)
    segments = len(unique_labels)
    
    # Tạo phần header
    header = [
        "format ascii cr",
        f"date {datetime.now().strftime('%a %b %d %H:%M:%S %Y')}",
        f"image {image_name}",
        "user 1102",  # Giữ nguyên như file mẫu
        f"width {w}",
        f"height {h}",
        f"segments {segments}",
        "gray 0",
        "invert 0",
        "flipflop 0",
        "data"
    ]
    
    # Tạo dữ liệu pixel theo định dạng (nhãn, dòng, cột bắt đầu, cột kết thúc)
    data_lines = []
    for row in range(h):
        row_labels = labels[row, :]
        start_col = 0
        current_label = row_labels[0]
        
        for col in range(1, w):
            if row_labels[col] != current_label:
                data_lines.append(f"{current_label} {row} {start_col} {col}")
                start_col = col
                current_label = row_labels[col]
        
        # Thêm dòng cuối cùng của hàng
        data_lines.append(f"{current_label} {row} {start_col} {w}")
    
    # Lưu vào file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n")
        f.write("\n".join(data_lines) + "\n")
    
    print(f"✅ File SEG đã lưu: {output_path}")



def normalized_cuts(lan, imagename, image_path):
    start_cpu = time.time()
    image = io.imread(image_path)
    image = color.gray2rgb(image) if image.ndim == 2 else image[:, :, :3] if image.shape[2] == 4 else image
    image = image / 255.0
    k = 2
    
    W, W_f, W_c = compute_weight_matrix(image)
    W_all = W_f + W_c
    
    L, D = compute_laplacian(W)
    vecs, lanczos_time = compute_eigen(L, D, k)
    labels = assign_labels(vecs, k)

    # Lưu kết quả dưới dạng file SEG
    seg_output_path = os.path.splitext(image_path)[0] + ".seg"
    save_seg_file(labels.reshape(image.shape[:2]), image.shape, seg_output_path, imagename)
    
    end_cpu = time.time()
    total_cpu_time = end_cpu - start_cpu
    
    return total_cpu_time, W_f, W_c, W_all
