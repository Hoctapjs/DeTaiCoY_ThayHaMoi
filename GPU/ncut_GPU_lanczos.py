import cupy as cp  # Thay thế NumPy bằng CuPy
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
# from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from skimage import io, color
import cupyx.scipy.sparse as sp
from cupyx.scipy.sparse.linalg import eigsh
from cupyx.scipy.sparse import diags
import time
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import logging


def kiemThuChayNhieuLan(i, name):
        temp_chuoi = f"{name}{i}"
        temp_chuoi = temp_chuoi + '.txt'
        logging.basicConfig(filename = temp_chuoi, level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
        # Duong dan toi anh cua ban
        # Mở hộp thoại chọn ảnh
        image_path = "apple3_60x60.jpg"  # Thay bang duong dan anh cua ban
        """ image_path = "apple4_98x100.jpg"  # Thay bang duong dan anh cua ban """
        normalized_cuts(image_path, k=3)
        # image_path = open_file_dialog()
        # if image_path:
        #     logging.info(f"Da chon anh: {image_path}")
        #     normalized_cuts(image_path, k=3)  # Phan vung thanh 3 nhom
        # else:
        #     logging.info("Khong co anh nao duoc chon.")

# def compute_weight_matrix(image, sigma_i=0.1, sigma_x=10, window_size=100):
#     h, w, c = image.shape
#     logging.info(f"Kich thuoc anh: {h}x{w}x{c}")

#     coords = cp.array(cp.meshgrid(cp.arange(h), cp.arange(w))).reshape(2, -1).T

#     features = cp.array(image).reshape(-1, c)

#     logging.info(f"Kich thuoc dac trung mau: {features.shape}, Kich thuoc toa do: {coords.shape}")
#     logging.info(f"Đac trung mau:\n{features[:9, :9]}")
#     logging.info(f"Toa do:\n{coords[:9, :9]}")

#     W = cp.zeros((h * w, h * w), dtype=cp.float32)  # Khoi tao ma tran trong so
#     for i in range(0, h * w, window_size):
#         end = min(i + window_size, h * w)
#         # Tinh toan tren phan nho du lieu
#         local_weights = cp.array(rbf_kernel(features[i:end].get(), features.get(), gamma=1/(2 * sigma_i**2))) * \
#                         cp.array(rbf_kernel(coords[i:end].get(), coords.get(), gamma=1/(2 * sigma_x**2)))
#         W[i:end, :] = local_weights

#     logging.info(f"Manh cua W (9x9 phan tu dau):\n{W[:9, :9]}")
#     return W

# ĐÂY LÀ CÁCH CHẠY MA TRẬN TRỌNG SỐ W TRÊN GPU THEO LOGIC GIỐNG BÊN CPU
def compute_weight_matrix(image, sigma_i=0.1, sigma_x=10):
    h, w, c = image.shape
    logging.info(f"Kích thước ảnh: {h}x{w}x{c}")

    # Tọa độ (x, y)
    coords = cp.array(cp.meshgrid(cp.arange(h), cp.arange(w))).reshape(2, -1).T

    # Đặc trưng màu
    features = cp.array(image).reshape(-1, c)

    logging.info(f"Kích thước đặc trưng màu: {features.shape}, Kích thước tọa độ: {coords.shape}")
    logging.info(f"Đặc trưng màu (9 phần tử đầu):\n{features[:9, :9]}")
    logging.info(f"Tọa độ (9 phần tử đầu):\n{coords[:9, :9]}")

    # Tính ma trận trọng số bằng vector hóa
    W_color = cp.array(rbf_kernel(features.get(), gamma=1 / (2 * sigma_i**2)))
    W_space = cp.array(rbf_kernel(coords.get(), gamma=1 / (2 * sigma_x**2)))
    W = W_color * W_space

    logging.info(f"Mảnh của W (9x9 phần tử đầu):\n{W[:9, :9]}")
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
# def compute_eigen(L, D, k=2):
#     # Chuyen du lieu ve CPU vi eigsh chua ho tro GPU
#     L_cpu, D_cpu = L.get(), D.get()
#     vals, vecs = eigsh(L_cpu, k=k, M=D_cpu, which='SM')  # 'SM' tim tri rieng nho nhat
#     return cp.array(vecs)  # Tra ve k vector rieng (chuyen ve GPU)

def compute_eigen(L, D, k=2):
    """
    Giai bai toan tri rieng bang thuat toan Lanczos (eigsh) tren GPU.
    :param L: Ma tran Laplace thua (CuPy sparse matrix).
    :param D: Ma tran duong cheo (CuPy sparse matrix).
    :param k: So tri rieng nho nhat can tinh.
    :return: Cac vector rieng tuong ung (k vector).
    """
    # Chuan hoa ma tran Laplace: D^-1/2 * L * D^-1/2
    D_diag = D.diagonal()  # Lay duong cheo cua D
    D_diag[D_diag < 1e-10] = 1e-10  # Trahn chia cho 0 hoac gan 0
    D_inv_sqrt = diags(1.0 / cp.sqrt(D_diag))  # Tinh D^-1/2
    L_normalized = D_inv_sqrt @ L @ D_inv_sqrt  # Chuan hoa ma tran Laplace

    # Giai bai toan tri rieng bang eigsh
    eigvals, eigvecs = eigsh(L, k=k, which='SA')  # Dung SA thay vi SM

    # Chuyen lai eigenvectors ve khong gian goc bang cach nhan D^-1/2
    eigvecs_original = D @ eigvecs

    return eigvecs_original

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
    start_gpu = time.time()
    
    # Doc anh va chuan hoa
    image = io.imread(image_path)
    if image.ndim == 2:  # Neu la anh xam, chuyen thanh RGB
        image = color.gray2rgb(image)
    elif image.shape[2] == 4:  # Neu la anh RGBA, loai bo kenh alpha
        image = image[:, :, :3]
    image = image / 255.0  # Chuan hoa ve [0, 1]
    
    # Tinh toan Ncuts
    logging.info("Tinh ma tran trong so...")
    W = compute_weight_matrix(image)
    
    logging.info("Tinh Laplace...")
    L, D = compute_laplacian(W)
    
    logging.info("Tinh eigenvectors...")
    eigen_vectors = compute_eigen(L, D, k=k)  # Tinh k vector rieng
    
    logging.info("Phan vung do thi...")
    labels = assign_labels(eigen_vectors, k)  # Gan nhan cho moi diem anh
    
    logging.info("Hien thi ket qua...")

    cp.cuda.Stream.null.synchronize()  # Dong bo hoa de dam bao GPU hoan thanh tinh toan
    end_gpu = time.time()
    logging.info(f"Thoi gian: {end_gpu - start_gpu} giay")

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
