import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from skimage import io, color
import time
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import logging
from scipy.sparse import coo_matrix #chuyển sang ma trận coo
from scipy.signal import find_peaks
from scipy.sparse import diags



def kiemThuChayNhieuLan(i, name):
        temp_chuoi = f"{name}{i}"
        temp_chuoi = temp_chuoi + '.txt'
        logging.basicConfig(filename = temp_chuoi, level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
        # Duong dan toi anh cua ban
        image_path = "apple3_60x60.jpg"  # Thay bang duong dan anh cua ban
        """ image_path = "apple4_98x100.jpg"  # Thay bang duong dan anh cua ban """
        normalized_cuts(image_path, k=3)

        # Mở hộp thoại chọn ảnh
        # image_path = open_file_dialog()
        # if image_path:
        #     logging.info(f"Da chon anh: {image_path}")
        #     normalized_cuts(image_path, k=3)  # Phan vung thanh 3 nhom
        # else:
        #     logging.info("Khong co anh nao duoc chon.")

    
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

# 1. Tinh ma tran trong so
def compute_weight_matrix(image, sigma_i=0.1, sigma_x=10):
    h, w, c = image.shape
    coords = np.array(np.meshgrid(range(h), range(w))).reshape(2, -1).T  # Toa do (x, y)
    features = image.reshape(-1, c)  # Dac trung mau
    
    logging.info(f"Kich thuoc anh: {h}x{w}x{c}")
    logging.info(f"Kich thuoc dac trung mau: {features.shape}, Kich thuoc toa do: {coords.shape}")
    logging.info(f"Dac trung mau:\n{features[:9, :9]}")
    logging.info(f"Toa do:\n{coords[:9, :9]}")

    
    # Tinh do tuong dong ve dac trung va khong gian
    W_features = rbf_kernel(features, gamma=1/(2 * sigma_i**2))
    W_coords = rbf_kernel(coords, gamma=1/(2 * sigma_x**2))
    W = W_features * W_coords
    
    logging.info(f"Kich thuoc ma tran trong so (W): {W.shape}")
    logging.info(f"Kich thuoc ma tran dac trung mau: {W_features.shape}, Kich thuoc ma tran toa do: {W_coords.shape}")
    logging.info(f"Mau cua W_features (9x9 phan tu dau):\n{W_features[:9, :9]}")
    logging.info(f"Mau cua W_coords (9x9 phan tu dau):\n{W_coords[:9, :9]}")
    logging.info(f"Mau cua W (9x9 phan tu dau):\n{W[:9, :9]}")

    # Chuyen ma tran W sang dang ma tran thua COO
    W_sparse = coo_matrix(W)
    logging.info(f"Kich thuoc ma tran thua (COO): {W_sparse.shape}")
    logging.info(f"Mau cua ma tran thua (COO) [du lieu, hang, cot]:\n{W_sparse.data[:9]}, {W_sparse.row[:9]}, {W_sparse.col[:9]}")
    
    return W_sparse



# 2. Tinh ma tran Laplace
def compute_laplacian(W_sparse):
    # Tạo ma trận đường chéo từ tổng các hàng
    D_diag = W_sparse.sum(axis=1).A.flatten() 
    D = np.diag(D_diag)  # Ma trận đường chéo
    L = D - W_sparse # L = D - W
    logging.info("Kich thuoc ma tran duong cheo (D): %s", D.shape)
    logging.info("Mau cua D (9 phan tu dau):\n%s", D_diag[:9])  # In phần tử trên đường chéo
    logging.info("Kich thuoc ma tran Laplace (L): %s", L.shape)
    logging.info("Mau cua L (9x9 phan tu dau):\n%s", L[:9, :9])
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
    D_inv_sqrt = diags(1.0 / np.sqrt(D_diag))  # Tinh D^-1/2
    L_normalized = D_inv_sqrt @ L @ D_inv_sqrt  # Chuan hoa ma tran Laplace

    # Giai bai toan tri rieng bang eigsh
    eigvals, eigvecs = eigsh(L_normalized, k=k, which='SA')  # Dung SA thay vi SM

    # Chuyen lai eigenvectors ve khong gian goc bang cach nhan D^-1/2
    eigvecs_original = D_inv_sqrt @ eigvecs

    return eigvecs_original


# 4. Gan nhan cho tung diem anh dua tren vector rieng
def assign_labels(eigen_vectors, k):
    # Dung K-Means de gan nhan
    kmeans = KMeans(n_clusters=k, random_state=0).fit(eigen_vectors)
    labels = kmeans.labels_
    logging.info(f"Nhan gan cho 27 pixel dau tien: {labels[:27]}")
    return labels

# 5. Hien thi ket qua
def display_segmentation(image, labels, k):
    h, w, c = image.shape
    segmented_image = np.zeros_like(image, dtype=np.uint8)
    
    # Tao bang mau ngau nhien
    colors = np.random.randint(0, 255, size=(k, 3), dtype=np.uint8)
    
    # To mau tung vung
    for i in range(k):
        segmented_image[labels.reshape(h, w) == i] = colors[i]
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Anh goc")
    plt.imshow(image)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Anh sau phan vung")
    plt.imshow(segmented_image)
    plt.axis('off')
    plt.show()

# 6. Ket hop toan bo
def normalized_cuts(image_path, k):
    # Tinh thoi gian tren CPU
    start_cpu = time.time()

    # Doc anh va chuan hoa
    image = io.imread(image_path)
    if image.ndim == 2:  # Neu la anh xam, chuyen thanh RGB
        image = color.gray2rgb(image)
    elif image.shape[2] == 4:  # Neu la anh RGBA, loai bo kenh alpha
        image = image[:, :, :3]
    image = image / 255.0  # Chuan hoa ve [0, 1]
    
      # Xác định số cụm k dựa trên histogram
    logging.info("Determining optimal k from histogram...")
    k = determine_max_k(image)
    logging.info(f"Optimal k determined: {k}")

    # Tinh toan Ncuts
    logging.info("Bat dau tinh ma tran trong so...")
    W = compute_weight_matrix(image)
    
    logging.info("Tinh ma tran Laplace...")
    L, D = compute_laplacian(W)
    
    logging.info("Tinh vector rieng...")
    eigen_vectors = compute_eigen(L, D, k=k)  # Tinh k vector rieng
    
    logging.info("Gan nhan cho cac diem anh...")
    labels = assign_labels(eigen_vectors, k)  # Gan nhan cho moi diem anh
    
    logging.info("Hien thi ket qua...")

    end_cpu = time.time()
    logging.info(f"Thoi gian: {end_cpu - start_cpu} giay")

    display_segmentation(image, labels, k)

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

