import cupy as cp  # Thay thế NumPy bằng CuPy
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from cupyx.scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from skimage import io, color
import cupyx.scipy.sparse as sp
import time
import logging
from scipy.signal import find_peaks
from cupyx.scipy.sparse import coo_matrix
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from  cupyx.scipy.sparse import diags
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
            f.write(f"{img}: {avg_time_per_image[img]:.4f} giây (Tổng) | {avg_coo_time_per_image[img]:.4f} giây (COO)\n")

        f.write(f"\nThời gian trung bình của cả thư mục:\n")
        f.write(f"Tổng: {avg_time_folder:.4f} giây\n")
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
    """Tính toán ma trận trọng số W sử dụng CuPy và CUDA Streams."""
    
    h, w, c = image.shape
    
    # Tạo tọa độ điểm ảnh
    coords = cp.array(cp.meshgrid(cp.arange(h), cp.arange(w))).reshape(2, -1).T
    features = image.reshape(-1, c)

    # Khởi tạo CUDA Streams
    stream1 = cp.cuda.Stream(non_blocking=True)
    stream2 = cp.cuda.Stream(non_blocking=True)

    # Tính toán W_features và W_coords song song
    with stream1: # song song 
        W_features = rbf_kernel(cp.asnumpy(features), gamma=1/(2 * sigma_i**2))
        W_features = cp.asarray(W_features)

    with stream2: 
        W_coords = rbf_kernel(cp.asnumpy(coords), gamma=1/(2 * sigma_x**2))
        W_coords = cp.asarray(W_coords)

    # Đồng bộ hóa Streams
    stream1.synchronize()
    stream2.synchronize()

    # Nhân ma trận để tạo W
    W = cp.multiply(W_features, W_coords)

    # Chuyển sang dạng ma trận thưa (sparse matrix)
    W_sparse = coo_matrix(W)

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
def compute_eigen(L, D, k=2):
    """
    Giai bai toan tri rieng bang thuat toan Lanczos (eigsh) tren GPU.
    :param L: Ma tran Laplace thua (CuPy sparse matrix).
    :param D: Ma tran duong cheo (CuPy sparse matrix).
    :param k: So tri rieng nho nhat can tinh.
    :return: Cac vector rieng tuong ung (k vector).
    """
    # Chuan hoa ma tran Laplace: D^-1/2 * L * D^-1/2
    D_diag = D.diagonal().copy()  # Lay duong cheo cua D
    D_diag[D_diag < 1e-10] = 1e-10  # Tranh chia cho 0 hoac gan 0
    D_inv_sqrt = diags(1.0 / cp.sqrt(D_diag))  # Tinh D^-1/2
    L_normalized = D_inv_sqrt @ L @ D_inv_sqrt  # Chuan hoa ma tran Laplace

    # Giai bai toan tri rieng bang eigsh
    eigvals, eigvecs = eigsh(L_normalized, k=k, which='SA')  # Dung SA thay vi SM

    # Chuyen lai eigenvectors ve khong gian goc bang cach nhan D^-1/2
    eigvecs_original = D_inv_sqrt @ eigvecs

    return eigvecs_original

# 4. Gan nhan cho tung diem anh dua tren vector rieng
def assign_labels(eigen_vectors, k):
    # Chuyen du lieu ve CPU de dung K-Means
    eigen_vectors_cpu = eigen_vectors.get()
    # logging.info(f"Mau cua vector rieng (9 hang dau):\n{eigen_vectors_cpu[:9, :]}")

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
    start_cpu_coo = time.time()
    # logging.info("Dang tinh toan ma tran trong so...")
    W = compute_weight_matrix(image)
    end_cpu_coo = time.time()

    # logging.info("Dang tinh toan ma tran Laplace...")
    L, D = compute_laplacian(W)
    
    # logging.info("Dang tinh vector rieng...")
    eigen_vectors = compute_eigen(L,D, k=k)  # Tinh k vector rieng
    
    # logging.info("Dang phan vung do thi...")
    labels = assign_labels(eigen_vectors, k)  # Gan nhan cho moi diem anh
    
    # logging.info("Dang hien thi ket qua...")

    cp.cuda.Stream.null.synchronize()  # Dong bo hoa de dam bao GPU hoan thanh tinh toan
    end_gpu = time.time()
    logging.info(f"Thoi gian COO: {end_cpu_coo - start_cpu_coo} giay")
    logging.info(f"Thoi gian: {end_gpu - start_gpu} giay")

    # display_segmentation(image, labels, k)
    
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
