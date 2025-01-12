import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from skimage import io, color
from sklearn.metrics import silhouette_score

# Xác định giá trị k_max tự động
def determine_max_k(image, sigma_i=0.1, sigma_x=10):
    # Tính ma trận trọng số
    W = compute_weight_matrix(image, sigma_i, sigma_x)
    L, D = compute_laplacian(W)

    # Phân tích trị riêng
    vals, _ = eigsh(L, k=min(20, L.shape[0]-2), M=D, which='SM')  # Lấy tối đa 20 trị riêng

    # Dựa trên khoảng cách trị riêng
    optimal_k = determine_optimal_eigenvalues(vals)

    # Giới hạn k_max dựa trên kích thước hình ảnh
    h, w, _ = image.shape
    max_k = min(optimal_k + 5, int(np.sqrt(h * w) / 10))
    return max(2, max_k)  # Đảm bảo k_max >= 2

# Xác định số cụm tối ưu dựa trên Silhouette score
def determine_optimal_clusters(eigen_vectors, max_k):
    best_k = 2
    best_score = -1
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(eigen_vectors)
        score = silhouette_score(eigen_vectors, kmeans.labels_)
        if score > best_score:
            best_k = k
            best_score = score
    return best_k

# Tính số vector riêng cần thiết dựa trên Eigenvalue Gap
def determine_optimal_eigenvalues(vals):
    gaps = np.diff(vals)
    optimal_k = np.argmax(gaps) + 1  # Tìm khoảng cách lớn nhất
    return optimal_k



# 1. Tính ma trận trọng số
def compute_weight_matrix(image, sigma_i=0.1, sigma_x=10):
    h, w, c = image.shape
    coords = np.array(np.meshgrid(range(h), range(w))).reshape(2, -1).T  # Tọa độ (x, y)
    features = image.reshape(-1, c)  # Đặc trưng màu
    
    # Tính độ tương đồng về đặc trưng và không gian
    W = rbf_kernel(features, gamma=1/(2 * sigma_i**2)) * rbf_kernel(coords, gamma=1/(2 * sigma_x**2))
    return W

# 2. Tính ma trận Laplace
def compute_laplacian(W):
    D = np.diag(W.sum(axis=1))  # Ma trận đường chéo
    L = D - W
    return L, D

# 3. Giải bài toán trị riêng
def compute_eigen(L, D, k=2):
    # Giải bài toán trị riêng tổng quát
    vals, vecs = eigsh(L, k=k, M=D, which='SM')  # 'SM' tìm trị riêng nhỏ nhất
    return vecs  # Trả về k vector riêng

# 4. Gán nhãn cho từng điểm ảnh dựa trên vector riêng
def assign_labels(eigen_vectors, k):
    # Dùng K-Means để gán nhãn
    kmeans = KMeans(n_clusters=k, random_state=0).fit(eigen_vectors)
    labels = kmeans.labels_
    return labels

""" # 5. Hiển thị kết quả
def display_segmentation(image, labels, k):
    h, w, c = image.shape
    segmented_image = np.zeros_like(image, dtype=np.uint8)
    
    # Tạo bảng màu ngẫu nhiên
    colors = np.random.randint(0, 255, size=(k, 3), dtype=np.uint8)
    
    # Tô màu từng vùng
    for i in range(k):
        segmented_image[labels.reshape(h, w) == i] = colors[i]
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Segmented Image")
    plt.imshow(segmented_image)
    plt.axis('off')
    plt.show() """

def display_segmentation(image, labels, k):
    """
    Hiển thị kết quả phân đoạn ảnh.

    Args:
        image (numpy.ndarray): Ảnh gốc, kích thước (h, w, c).
        labels (numpy.ndarray): Nhãn của từng pixel, kích thước (h * w,).
        k (int): Số cụm.
    """
    h, w, c = image.shape
    segmented_image = np.zeros_like(image, dtype=np.uint8)

    # Tạo bảng màu ngẫu nhiên
    colors = np.random.randint(0, 255, size=(k, 3), dtype=np.uint8)

    # Tô màu từng vùng theo màu trung bình
    for i in range(k):
        mask = labels.reshape(h, w) == i  # Mảng boolean xác định vùng thuộc cụm i
        cluster_pixels = image[mask]  # Lấy tất cả pixel thuộc cụm i

        if len(cluster_pixels) > 0:
            # Tính màu trung bình cho cụm i
            mean_color = (cluster_pixels.mean(axis=0) * 255).astype(np.uint8)
        else:
            # Nếu cụm không có pixel nào, gán màu đen
            mean_color = np.array([0, 0, 0], dtype=np.uint8)

        # Gán màu trung bình vào vùng cụm i
        segmented_image[mask] = mean_color

    # Hiển thị ảnh gốc và ảnh sau phân đoạn
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow((image * 255).astype(np.uint8))  # Chuyển ảnh về [0, 255] để hiển thị đúng
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Segmented Image")
    plt.imshow(segmented_image)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# 6. Kết hợp toàn bộ
def normalized_cuts(image_path):
    # Đọc ảnh và chuẩn hóa
    image = io.imread(image_path)
    if image.ndim == 2:  # Nếu là ảnh xám, chuyển thành RGB
        image = color.gray2rgb(image)
    elif image.shape[2] == 4:  # Nếu là ảnh RGBA, loại bỏ kênh alpha
        image = image[:, :, :3]
    image = image / 255.0  # Chuẩn hóa về [0, 1]

    # Xác định giá trị k_max tự động
    print("Determining optimal k_max...")
    max_k = determine_max_k(image)

    # Tính toán Ncuts
    print("Computing weight matrix...")
    W = compute_weight_matrix(image)
    
    print("Computing Laplacian...")
    L, D = compute_laplacian(W)

    print("Computing eigenvalues and eigenvectors...")
    vals, vecs = eigsh(L, k=max_k, M=D, which='SM')  # Lấy nhiều vector riêng để phân tích

    print("Determining optimal number of clusters...")
    optimal_eigen_k = determine_optimal_eigenvalues(vals)
    eigen_vectors = vecs[:, :optimal_eigen_k]  # Chỉ lấy các vector riêng cần thiết

    print("Determining optimal k for clustering...")
    optimal_clusters = determine_optimal_clusters(eigen_vectors, max_k=max_k)

    print(f"Optimal clusters: {optimal_clusters}, Eigen vectors used: {optimal_eigen_k}")

    print("Partitioning graph...")
    labels = assign_labels(eigen_vectors, optimal_clusters)

    print("Displaying results...")
    display_segmentation(image, labels, optimal_clusters)

# 7. Chạy thử nghiệm
if __name__ == "__main__":
    # image_path = "D:/result_traitao.jpg"  # Đường dẫn ảnh tải lên
    # image_path = "D:/cayco.png"  # Đường dẫn ảnh tải lên
    # image_path = "D:/result_profile.png"  # Đường dẫn ảnh tải lên
    # image_path = "D:/result_options.png"  # Đường dẫn ảnh tải lên
    image_path = "apple3_60x60.jpg"  # Đường dẫn ảnh tải lên
    normalized_cuts(image_path)