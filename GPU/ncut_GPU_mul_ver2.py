import cupy as cp  # Thay thế NumPy bằng CuPy
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
# from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from skimage import io, color
import cupyx.scipy.sparse as sp
from cupyx.scipy.sparse.linalg import eigsh
import time
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import logging
from cupyx.scipy.sparse import coo_matrix
import concurrent.futures
import threading




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


# 1. Tinh ma tran trong so
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
    with stream1:
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
    D_diag = W_sparse.sum(axis=1).flatten()  # Tính tổng các hàng
    D = cp.diag(D_diag)  # Tạo ma trận đường chéo từ tổng hàng
    L = D - W_sparse  # L = D - W
    logging.info("Kich thuoc ma tran duong cheo (D): %s", D.shape)
    logging.info("Mau cua D (9 phan tu dau):\n%s", D_diag[:9])  # In phần tử trên đường chéo
    logging.info("Kich thuoc ma tran Laplace (L): %s", L.shape)
    logging.info("Mau cua L (9x9 phan tu dau):\n%s", L[:9, :9])
    return L, D



# 3. Giai bai toan tri rieng
def compute_eigen(L, k=2):
    
    # Tìm các trị riêng nhỏ nhất (Smallest Magnitude)
    eigvals, eigvecs = eigsh(L, k=k, which='SA')  
    return eigvecs





# 4. Gan nhan cho tung diem anh duoc dua tren vector rieng
def assign_labels(eigen_vectors, k):
    # Chuyen du lieu ve CPU de dung K-Means
    eigen_vectors_cpu = eigen_vectors.get()
    logging.info(f"Manh cua vector rieng (9 hang dau):\n{eigen_vectors_cpu[:9, :]}")

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
    # Tinh tong tren GPU
    # start_gpu = time.time()
    # Doc anh va chuan hoa
    image = io.imread(image_path)
    if image.ndim == 2:  # Neu la anh xam, chuyen thanh RGB
        image = color.gray2rgb(image)
    elif image.shape[2] == 4:  # Neu la anh RGBA, loai bo kenh alpha
        image = image[:, :, :3]
    image = image / 255.0  # Chuan hoa ve [0, 1]
    

    # Tính thời gian riêng cho COO matrix
    start_cpu_coo = time.time()
    start_W = time.time()
    # Tinh toan Ncuts
    logging.info("Tinh ma tran trong so...")
    W_sparse = compute_weight_matrix(image)  # Ma trận W ở dạng thưa
    end_cpu_coo = time.time()
    end_W = time.time()
    
    logging.info("Tinh Laplace...")
    start_L = time.time()
    L, D = compute_laplacian(W_sparse)
    end_L = time.time()
    
    logging.info("Tinh eigenvectors...")
    start_Eigen = time.time()
    eigen_vectors = compute_eigen(L, k=k)  # Tinh k vector rieng
    end_Eigen = time.time()

    logging.info("Phan vung do thi...")
    labels = assign_labels(eigen_vectors, k)  # Gan nhan cho moi diem anh
    
    logging.info("Hien thi ket qua...")

    # Đồng bộ hóa tất cả các luồng GPU trước khi hiển thị kết quả
    cp.cuda.Device(0).synchronize()
    # end_gpu = time.time()
    # logging.info(f"Thoi gian: {end_gpu - start_gpu} giay")
    logging.info(f"Thoi gian COO: {end_cpu_coo - start_cpu_coo} giay")

    total_time = end_Eigen - start_W
    logging.info(f"Time W: {end_W - start_W} s")
    logging.info(f"Time L: {end_L - start_L} s")
    logging.info(f"Time Eigen: {end_Eigen - start_Eigen} s")
    logging.info(f"Total Time: {total_time} s")

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


