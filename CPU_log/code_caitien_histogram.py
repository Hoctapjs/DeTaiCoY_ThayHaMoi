import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from skimage import io, color
from scipy.signal import find_peaks
import time
import logging

def kiemThuChayNhieuLan(i, name):
        temp_chuoi = f"{name}{i}"
        temp_chuoi = temp_chuoi + '.txt'
        logging.basicConfig(filename = temp_chuoi, level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
        # Duong dan toi anh cua ban
        image_path = "apple3_60x60.jpg"  # Thay bang duong dan anh cua ban
        """ image_path = "apple4_98x100.jpg"  # Thay bang duong dan anh cua ban """
        normalized_cuts(image_path)

def find_peaks_with_conditions(histogram, delta_threshold, dist_threshold):
    """
    Tìm các đỉnh cực đại trong histogram thỏa mãn điều kiện về độ lệch và khoảng cách
    
    Args:
        histogram: Mảng histogram
        delta_threshold: Ngưỡng độ lệch chiều cao tối thiểu (sigma*)
        dist_threshold: Ngưỡng khoảng cách tối thiểu (delta*)
    
    Returns:
        peaks: Các vị trí của đỉnh cực đại thỏa mãn điều kiện
    """
    # Tìm tất cả các đỉnh cực đại
    peaks, _ = find_peaks(histogram)
    
    # Lọc các đỉnh theo điều kiện
    valid_peaks = []
    
    for i in range(len(peaks)):
        is_valid = True
        for j in range(len(peaks)):
            if i != j:
                # Tính độ lệch chiều cao và khoảng cách
                delta = abs(histogram[peaks[i]] - histogram[peaks[j]])
                dist = abs(peaks[i] - peaks[j])
                
                # Kiểm tra điều kiện theo công thức (3.1) và (3.2)
                if delta < delta_threshold or dist < dist_threshold:
                    is_valid = False
                    break
        
        if is_valid:
            valid_peaks.append(peaks[i])
    
    return np.array(valid_peaks)

def determine_k_from_histogram(image):
    """
    Xác định số nhóm k dựa trên phân tích histogram
    
    Args:
        image: Ảnh đầu vào (đã chuẩn hóa về [0, 1])
        
    Returns:
        k: Số nhóm cần phân đoạn
    """
    # Chuyển ảnh sang ảnh xám nếu là ảnh màu
    if len(image.shape) == 3:
        gray_image = color.rgb2gray(image)
    else:
        gray_image = image
    
    # Tính histogram
    histogram, _ = np.histogram(gray_image, bins=256, range=(0, 1))
    
    # Các tham số ngưỡng (cần điều chỉnh dựa trên tập huấn luyện)
    delta_threshold = np.max(histogram) * 0.1  # sigma* = 10% của giá trị cao nhất
    dist_threshold = 20  # delta* = 20 bins
    
    # Tìm các đỉnh thỏa mãn điều kiện
    valid_peaks = find_peaks_with_conditions(histogram, delta_threshold, dist_threshold)
    
    # Số nhóm k là số đỉnh hợp lệ
    k = len(valid_peaks)
    
    # Đảm bảo k ít nhất là 2
    return max(2, k)

def determine_max_k(image, sigma_i=0.1, sigma_x=10):
    """
    Xác định số nhóm k tối đa dựa trên histogram
    
    Args:
        image: Ảnh đầu vào
        sigma_i, sigma_x: Các tham số cho tính toán ma trận trọng số (không sử dụng trong phương pháp mới)
    
    Returns:
        k: Số nhóm tối đa cần phân đoạn
    """
    k = determine_k_from_histogram(image)
    
    # Giới hạn k dựa trên kích thước ảnh để tránh over-segmentation
    h, w, _ = image.shape
    max_k = min(k, int(np.sqrt(h * w) / 10))
    
    return max(2, max_k)

def compute_weight_matrix(image, sigma_i=0.1, sigma_x=10):
    """
    Tính ma trận trọng số W dựa trên đặc trưng màu và vị trí không gian
    """
    h, w, c = image.shape
    coords = np.array(np.meshgrid(range(h), range(w))).reshape(2, -1).T
    features = image.reshape(-1, c)
    
    W = rbf_kernel(features, gamma=1/(2 * sigma_i**2)) * rbf_kernel(coords, gamma=1/(2 * sigma_x**2))
    return W

def compute_laplacian(W):
    """
    Tính ma trận Laplace từ ma trận trọng số
    """
    D = np.diag(W.sum(axis=1))
    L = D - W
    return L, D

def compute_eigen(L, D, k=2):
    """
    Tính k vector riêng nhỏ nhất của bài toán trị riêng tổng quát
    """
    vals, vecs = eigsh(L, k=k, M=D, which='SM')
    return vals, vecs

def assign_labels(eigen_vectors, k):
    """
    Gán nhãn cho các pixel dựa trên vector riêng sử dụng K-Means
    """
    kmeans = KMeans(n_clusters=k, random_state=0).fit(eigen_vectors)
    return kmeans.labels_

def display_segmentation(image, labels, k):
    """
    Hiển thị kết quả phân đoạn ảnh
    """
    h, w, c = image.shape
    segmented_image = np.zeros_like(image, dtype=np.uint8)

    # Tính và gán màu trung bình cho từng vùng
    for i in range(k):
        mask = labels.reshape(h, w) == i
        cluster_pixels = image[mask]
        
        if len(cluster_pixels) > 0:
            mean_color = (cluster_pixels.mean(axis=0) * 255).astype(np.uint8)
        else:
            mean_color = np.array([0, 0, 0], dtype=np.uint8)
            
        segmented_image[mask] = mean_color

    # Hiển thị kết quả
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow((image * 255).astype(np.uint8))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"Segmented Image (k={k})")
    plt.imshow(segmented_image)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def normalized_cuts(image_path):
    """
    Hàm chính thực hiện phân đoạn ảnh sử dụng Normalized Cuts
    """
    # Tinh thoi gian tren CPU
    start_cpu = time.time()
    # Đọc và tiền xử lý ảnh
    print("Loading and preprocessing image...")
    image = io.imread(image_path)
    if image.ndim == 2:
        image = color.gray2rgb(image)
    elif image.shape[2] == 4:
        image = image[:, :, :3]
    image = image / 255.0

    # Xác định số cụm k dựa trên histogram
    print("Determining optimal k from histogram...")
    k = determine_max_k(image)
    print(f"Optimal k determined: {k}")

    # Tính toán ma trận trọng số và Laplacian
    print("Computing weight matrix...")
    W = compute_weight_matrix(image)
    
    print("Computing Laplacian matrix...")
    L, D = compute_laplacian(W)

    # Tính vector riêng và phân cụm
    print("Computing eigenvectors...")
    vals, vecs = compute_eigen(L, D, k=k)
    
    print("Assigning labels...")
    labels = assign_labels(vecs, k)

    # Hiển thị kết quả
    print("Displaying results...")
    display_segmentation(image, labels, k)

    end_cpu = time.time()
    logging.info(f"Thoi gian: {end_cpu - start_cpu} giay")
    
    return labels, k

""" if __name__ == "__main__":
    # Đường dẫn ảnh cần phân đoạn
    image_path = "apple3_60x60.jpg"
    labels, k = normalized_cuts(image_path)
    print(f"Segmentation completed with {k} segments") """