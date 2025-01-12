import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from skimage import io, color
import time



# 1. Tính ma trận trọng số
def compute_weight_matrix(image, sigma_i=0.1, sigma_x=10):
    h, w, c = image.shape
    coords = np.array(np.meshgrid(range(h), range(w))).reshape(2, -1).T  # Tọa độ (x, y)
    features = image.reshape(-1, c)  # Đặc trưng màu
    
    print(f"Kích thước ảnh: {h}x{w}x{c}")
    print(f"Kích thước đặc trưng màu: {features.shape}, Kích thước tọa độ: {coords.shape}")
    print(f"đặc trưng màu:\n{features[:9, :9]}")
    print(f"tọa độ:\n{coords[:9, :9]}")

    
    # Tính độ tương đồng về đặc trưng và không gian
    W_features = rbf_kernel(features, gamma=1/(2 * sigma_i**2))
    W_coords = rbf_kernel(coords, gamma=1/(2 * sigma_x**2))
    W = W_features * W_coords
    
    print(f"Kích thước ma trận trọng số (W): {W.shape}")
    print(f"Kích thước ma trận đặc trưng màu: {W_features.shape}, Kích thước ma trận tọa độ: {W_coords.shape}")
    print(f"Mẫu của W_features (9x9 phần tử đầu):\n{W_features[:9, :9]}")
    print(f"Mẫu của W_coords (9x9 phần tử đầu):\n{W_coords[:9, :9]}")
    print(f"Mẫu của W (9x9 phần tử đầu):\n{W[:9, :9]}")
    
    
    return W

# 2. Tính ma trận Laplace
def compute_laplacian(W):
    D = np.diag(W.sum(axis=1))  # Ma trận đường chéo
    L = D - W
    
    print("Kích thước ma trận đường chéo (D):", D.shape)
    print("Mẫu của D (9x9 phần tử đầu):\n", D[:9, :9])
    print("Kích thước ma trận Laplace (L):", L.shape)
    print("Mẫu của L (9x9 phần tử đầu):\n", L[:9, :9])
    
    return L, D

# 3. Giải bài toán trị riêng
def compute_eigen(L, D, k):
    # Giải bài toán trị riêng tổng quát
    vals, vecs = eigsh(L, k=k, M=D, which='SM')  # 'SM' tìm trị riêng nhỏ nhất
    
    print(f"Trị riêng (Eigenvalues): {vals}")
    print(f"Kích thước vector riêng: {vecs.shape}")
    print(f"Mẫu của vector riêng (9 hàng đầu):\n{vecs[:9, :]}")
    
    return vecs  # Trả về k vector riêng

# 4. Gán nhãn cho từng điểm ảnh dựa trên vector riêng
def assign_labels(eigen_vectors, k):
    # Dùng K-Means để gán nhãn
    kmeans = KMeans(n_clusters=k, random_state=0).fit(eigen_vectors)
    labels = kmeans.labels_
    
    print(f"Nhãn gán cho 27 pixel đầu tiên: {labels[:27]}")
    return labels

# 5. Hiển thị kết quả
def display_segmentation(image, labels, k):
    h, w, c = image.shape
    segmented_image = np.zeros_like(image, dtype=np.uint8)
    
    # Tạo bảng màu ngẫu nhiên
    colors = np.random.randint(0, 255, size=(k, 3), dtype=np.uint8)
    
    # Tô màu từng vùng
    for i in range(k):
        segmented_image[labels.reshape(h, w) == i] = colors[i]
    
    """ # Tính màu trung bình cho từng cụm
    for i in range(k):
        cluster_pixels = image[labels.reshape(h, w) == i]
        if len(cluster_pixels) > 0:
            mean_color = cluster_pixels.mean(axis=0)
        else:
            mean_color = np.array([0, 0, 0])
        segmented_image[labels.reshape(h, w) == i] = mean_color """
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Ảnh gốc")
    plt.imshow(image)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Ảnh sau phân vùng")
    plt.imshow(segmented_image)
    plt.axis('off')
    plt.show()

# 6. Kết hợp toàn bộ
def normalized_cuts(image_path, k):
    # Tính thời gian trên CPU
    start_cpu = time.time()

    # Đọc ảnh và chuẩn hóa
    image = io.imread(image_path)
    if image.ndim == 2:  # Nếu là ảnh xám, chuyển thành RGB
        image = color.gray2rgb(image)
    elif image.shape[2] == 4:  # Nếu là ảnh RGBA, loại bỏ kênh alpha
        image = image[:, :, :3]
    image = image / 255.0  # Chuẩn hóa về [0, 1]
    
    # Tính toán Ncuts
    print("Bắt đầu tính ma trận trọng số...")
    W = compute_weight_matrix(image)
    
    print("Tính ma trận Laplace...")
    L, D = compute_laplacian(W)
    
    print("Tính vector riêng...")
    eigen_vectors = compute_eigen(L, D, k=k)  # Tính k vector riêng
    
    print("Gán nhãn cho các điểm ảnh...")
    labels = assign_labels(eigen_vectors, k)  # Gán nhãn cho mỗi điểm ảnh
    
    print("Hiển thị kết quả...")

    end_cpu = time.time()
    print(f"thời gian: {end_cpu - start_cpu} giây")

    display_segmentation(image, labels, k)

    

# 7. Chạy thử nghiệm
if __name__ == "__main__":
    # Đường dẫn tới ảnh của bạn
    image_path = "apple3_60x60.jpg"  # Thay bằng đường dẫn ảnh của bạn
    normalized_cuts(image_path, k=3)  # Phân vùng thành 3 nhóm
