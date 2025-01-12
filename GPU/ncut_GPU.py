import cupy as cp  # Thay thế NumPy bằng CuPy
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from skimage import io, color
import cupyx.scipy.sparse as sp
import time


def compute_weight_matrix(image, sigma_i=0.1, sigma_x=10, window_size=100):
    h, w, c = image.shape
    print(f"Kích thước ảnh: {h}x{w}x{c}")

    coords = cp.array(cp.meshgrid(cp.arange(h), cp.arange(w))).reshape(2, -1).T

    features = cp.array(image).reshape(-1, c)

    print(f"Kích thước đặc trưng màu: {features.shape}, Kích thước tọa độ: {coords.shape}")
    print(f"đặc trưng màu:\n{features[:9, :9]}")
    print(f"tọa độ:\n{coords[:9, :9]}")

    W = cp.zeros((h * w, h * w), dtype=cp.float32)  # Khởi tạo ma trận trọng số
    for i in range(0, h * w, window_size):
        end = min(i + window_size, h * w)
        # Tính toán trên phần nhỏ dữ liệu
        local_weights = cp.array(rbf_kernel(features[i:end].get(), features.get(), gamma=1/(2 * sigma_i**2))) * \
                        cp.array(rbf_kernel(coords[i:end].get(), coords.get(), gamma=1/(2 * sigma_x**2)))
        W[i:end, :] = local_weights

    print(f"Mẫu của W (9x9 phần tử đầu):\n{W[:9, :9]}")
    return W




# 2. Tính ma trận Laplace
def compute_laplacian(W):
    W_sparse = sp.csr_matrix(W)  # Chuyển W thành ma trận thưa
    D_diag = W_sparse.sum(axis=1).get()  # Tính tổng các hàng
    D = sp.diags(D_diag.flatten())  # Tạo ma trận đường chéo từ tổng
    L = D - W_sparse  # L = D - W
    print("Kích thước ma trận đường chéo (D):", D.shape)
    print("Kích thước ma trận trọng số W:", W_sparse.shape)
    print("Kích thước ma trận Laplace (L):", L.shape)
    return L, D

# 3. Giải bài toán trị riêng
def compute_eigen(L, D, k=2):
    # Chuyển dữ liệu về CPU vì eigsh chưa hỗ trợ GPU
    L_cpu, D_cpu = L.get(), D.get()
    vals, vecs = eigsh(L_cpu, k=k, M=D_cpu, which='SM')  # 'SM' tìm trị riêng nhỏ nhất
    return cp.array(vecs)  # Trả về k vector riêng (chuyển về GPU)

# 4. Gán nhãn cho từng điểm ảnh dựa trên vector riêng
def assign_labels(eigen_vectors, k):
    # Chuyển dữ liệu về CPU để dùng K-Means
    eigen_vectors_cpu = eigen_vectors.get()
    print(f"Mẫu của vector riêng (9 hàng đầu):\n{eigen_vectors_cpu[:9, :]}")

    kmeans = KMeans(n_clusters=k, random_state=0).fit(eigen_vectors_cpu)
    labels = kmeans.labels_
    print(f"Nhãn gán cho 27 pixel đầu tiên: {labels[:27]}")
    return cp.array(labels)  # Chuyển lại về GPU

# 5. Hiển thị kết quả
def display_segmentation(image, labels, k):
    h, w, c = image.shape
    segmented_image = cp.zeros_like(cp.array(image), dtype=cp.uint8)
    
    # Tạo bảng màu ngẫu nhiên
    colors = cp.random.randint(0, 255, size=(k, 3), dtype=cp.uint8)
    
    # Tô màu từng vùng
    for i in range(k):
        segmented_image[labels.reshape(h, w) == i] = colors[i]
    
    # Hiển thị trên CPU
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

# 6. Kết hợp toàn bộ
def normalized_cuts(image_path, k=2):
    
    # Tính tổng trên GPU
    start_gpu = time.time()
    
    # Đọc ảnh và chuẩn hóa
    image = io.imread(image_path)
    if image.ndim == 2:  # Nếu là ảnh xám, chuyển thành RGB
        image = color.gray2rgb(image)
    elif image.shape[2] == 4:  # Nếu là ảnh RGBA, loại bỏ kênh alpha
        image = image[:, :, :3]
    image = image / 255.0  # Chuẩn hóa về [0, 1]
    
    # Tính toán Ncuts
    print("Computing weight matrix...")
    W = compute_weight_matrix(image)
    
    print("Computing Laplacian...")
    L, D = compute_laplacian(W)
    
    print("Computing eigenvectors...")
    eigen_vectors = compute_eigen(L, D, k=k)  # Tính k vector riêng
    
    print("Partitioning graph...")
    labels = assign_labels(eigen_vectors, k)  # Gán nhãn cho mỗi điểm ảnh
    
    print("Displaying results...")

    cp.cuda.Stream.null.synchronize()  # Đồng bộ hóa để đảm bảo GPU hoàn thành tính toán
    end_gpu = time.time()
    print(f"thời gian: {end_gpu - start_gpu} giây")

    display_segmentation(image, labels, k)

    

# 7. Chạy thử nghiệm
if __name__ == "__main__":
    # Đường dẫn tới ảnh của bạn
    image_path = "apple3_60x60.jpg"  # Thay bằng đường dẫn ảnh của bạn
    normalized_cuts(image_path, k=3)  # Phân vùng thành 4 nhóm
