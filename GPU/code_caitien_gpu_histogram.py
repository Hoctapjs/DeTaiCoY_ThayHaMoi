import cupy as cp  # Thay thế NumPy bằng CuPy
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from skimage import io, color
import cupyx.scipy.sparse as sp
import time
import logging
from scipy.signal import find_peaks


def kiemThuChayNhieuLan(i, name):
        temp_chuoi = f"{name}{i}"
        temp_chuoi = temp_chuoi + '.txt'
        logging.basicConfig(filename = temp_chuoi, level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
        # Duong dan toi anh cua ban
        # Mo hop thoai chon anh
        image_path = "apple3_60x60.jpg"  # Thay bang duong dan anh cua ban
        """ image_path = "apple4_98x100.jpg"  # Thay bang duong dan anh cua ban """
        normalized_cuts(image_path, k=3)


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
    """ peaks, _ = find_peaks(histogram) """
    peaks, _ = find_peaks(cp.asnumpy(histogram))  # SciPy hỗ trợ NumPy, cần chuyển về NumPy.
    peaks = cp.array(peaks)  # Chuyển lại về CuPy để tiếp tục xử lý trên GPU.
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
    
    return cp.array(valid_peaks)

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
        gray_image = cp.array(gray_image)  # Chuyển về GPU.
    else:
        gray_image = image
    
    # Tính histogram
    histogram, _ = cp.histogram(gray_image, bins=256, range=(0, 1))
    
    # Các tham số ngưỡng (cần điều chỉnh dựa trên tập huấn luyện)
    delta_threshold = cp.max(histogram) * 0.1  # sigma* = 10% của giá trị cao nhất
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
    max_k = min(k, int(cp.sqrt(h * w) / 10))
    
    return max(2, max_k)

def compute_weight_matrix(image, sigma_i=0.1, sigma_x=10):
    h, w, c = image.shape
    logging.info(f"Kich thuoc anh: {h}x{w}x{c}")

    # Toa do (x, y)
    coords = cp.array(cp.meshgrid(cp.arange(h), cp.arange(w))).reshape(2, -1).T

    # Dac trung mau
    features = cp.array(image).reshape(-1, c)

    logging.info(f"Kich thuoc dac trung mau: {features.shape}, Kich thuoc toa do: {coords.shape}")
    logging.info(f"Dac trung mau (9 phan tu dau):\n{features[:9, :9]}")
    logging.info(f"Toa do (9 phan tu dau):\n{coords[:9, :9]}")
    
    # Tinh ma tran trong so bang vector hoa
    W_color = cp.array(rbf_kernel(features.get(), gamma=1 / (2 * sigma_i**2)))
    W_space = cp.array(rbf_kernel(coords.get(), gamma=1 / (2 * sigma_x**2)))
    W = W_color * W_space

    logging.info(f"Manh cua W (9x9 phan tu dau):\n{W[:9, :9]}")
    return W

# 2. Tinh ma tran Laplace
def compute_laplacian(W):
    W_sparse = sp.csr_matrix(W)  # Chuyen W thanh ma tran thua
    D_diag = W_sparse.sum(axis=1).get()  # Tinh tong cac hang
    D = sp.diags(D_diag.flatten())  # Tao ma tran duong cheo tu tong
    L = D - W_sparse  # L = D - W
    logging.info("Kich thuoc ma tran duong cheo (D):", D.shape)
    logging.info("Kich thuoc ma tran trong so W:", W_sparse.shape)
    logging.info("Kich thuoc ma tran Laplace (L):", L.shape)
    return L, D

# 3. Giai bai toan tri rieng
def compute_eigen(L, D, k=2):
    # Chuyen du lieu ve CPU vi eigsh chua ho tro GPU
    L_cpu, D_cpu = L.get(), D.get()
    vals, vecs = eigsh(L_cpu, k=k, M=D_cpu, which='SM')  # 'SM' tim tri rieng nho nhat
    return cp.array(vecs)  # Tra ve k vector rieng (chuyen ve GPU)

# 4. Gan nhan cho tung diem anh dua tren vector rieng
def assign_labels(eigen_vectors, k):
    # Chuyen du lieu ve CPU de dung K-Means
    eigen_vectors_cpu = eigen_vectors.get()
    logging.info(f"Mau cua vector rieng (9 hang dau):\n{eigen_vectors_cpu[:9, :]}")

    kmeans = KMeans(n_clusters=k, random_state=0).fit(eigen_vectors_cpu)
    labels = kmeans.labels_
    logging.info(f"Nhan gan cho 27 pixel dau tien: {labels[:27]}")
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
    
    # Tinh toan tren GPU
    start_gpu = time.time()
    
    # Doc anh va chuan hoa
    image = io.imread(image_path)
    if image.ndim == 2:  # Neu la anh xam, chuyen thanh RGB
        image = color.gray2rgb(image)
    elif image.shape[2] == 4:  # Neu la anh RGBA, loai bo kenh alpha
        image = image[:, :, :3]
    image = image / 255.0  # Chuan hoa ve [0, 1]

    # Xác định số cụm k dựa trên histogram
    print("Determining optimal k from histogram...")
    k = determine_max_k(image)
    print(f"Optimal k determined: {k}")
    
    # Tinh toan Ncuts
    logging.info("Dang tinh toan ma tran trong so...")
    W = compute_weight_matrix(image)
    
    logging.info("Dang tinh toan ma tran Laplace...")
    L, D = compute_laplacian(W)
    
    logging.info("Dang tinh vector rieng...")
    eigen_vectors = compute_eigen(L, D, k=k)  # Tinh k vector rieng
    
    logging.info("Dang phan vung do thi...")
    labels = assign_labels(eigen_vectors, k)  # Gan nhan cho moi diem anh
    
    logging.info("Dang hien thi ket qua...")

    cp.cuda.Stream.null.synchronize()  # Dong bo hoa de dam bao GPU hoan thanh tinh toan
    end_gpu = time.time()
    logging.info(f"Thoi gian: {end_gpu - start_gpu} giay")

    display_segmentation(image, labels, k)
