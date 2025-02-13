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
    D = cp.diag(W.sum(axis=1))  # Ma trận đường chéo
    L = D - W
    logging.info("Kích thước ma trận đường chéo (D):", D.shape)
    logging.info("Mẫu của D (9x9 phần tử đầu):\n", D[:9, :9])
    logging.info("Kích thước ma trận Laplace (L):", L.shape)
    logging.info("Mẫu của L (9x9 phần tử đầu):\n", L[:9, :9])
    
    return L, D

# 3. Giai bai toan tri rieng

def Lanczos(A, v, m):
    """
    Thuật toán Lanczos để xấp xỉ trị riêng và vector riêng.
    : A: Ma trận cần tính (numpy 2D array).
    : v: Vector khởi tạo.
    : m: Số bước lặp Lanczos.
    :return: Ma trận tam giác T và ma trận trực giao V.
    """
    n = len(v) # Đây là số phần tử trong vector v (số chiều của ma trận A)
    V = cp.zeros((m, n)) # đây là một ma trận mxn lưu trữ các vector trực giao (là 2 vector có tích vô hướng = 0), mỗi hàng là một bước đã đi qua, np.zeros nghĩa là ban đầu tất cả các bước đi (hay các phần tử của ma trận) đều là 0, chưa đi bước nào
    T = cp.zeros((m, m)) # đây là ma trận tam giác T
    V[0, :] = v / cp.linalg.norm(v) # np.linalg.norm(v) là để tính chuẩn (độ dài) của vector = căn(v1^2 + v2^2 + ...)
    # => V[0, :] = v / np.linalg.norm(v) là để chuẩn hóa vector v đầu vào thành vector đơn vị 
    
    # Đoạn này là để làm cho w trực giao với V0 thôi
    # vd: để làm cho 2 vector a và b trực giao với nhau
    # 1. tính tích vô hướng của a và b (alpha)
    # 2. cập nhật vector a lại 
    #   a = a - alpha * b (b ở đây là V[0, :] = v / căn(v) )


    w = A @ V[0, :] # tính vector w bằng cách nhân A với vector đầu tiên của V - hiểu nôm na là w sẽ cho ta biết các mà ma trận A tương tác với vector khởi tạo v
    alpha = cp.dot(w, V[0, :]) # .dot là tính tích vô hướng của 2 vector a và b (trong case này là w và vector đầu tiên của V), hệ số alpha là để đo mức độ song song giữa w và V0
    w = w - alpha * V[0, :]
    # alpha * V[0, :] tạo ra một vector có hướng song song với 
    # V[0,:] mà có độ dài tương ứng.
    # sau khi trừ xong thì nò sẽ loại bỏ phần song song ra khỏi w

    
    T[0, 0] = alpha # Gán giá trị alpha vào phần tử đầu tiên của T
    
    for j in range(1, m):
        beta = cp.linalg.norm(w)
        if beta < 1e-10:
            break
        V[j, :] = w / beta
        w = A @ V[j, :]
        alpha = cp.dot(w, V[j, :])
        w = w - alpha * V[j, :] - beta * V[j-1, :]
        
        T[j, j] = alpha
        T[j-1, j] = beta
        T[j, j-1] = beta
    
    return T, V

def compute_eigen(L, D, k=2):
    """
    Giải bài toán trị riêng bằng thuật toán Lanczos không dùng eigsh.
    :param L: Ma trận Laplace thưa (Scipy sparse matrix).
    :param D: Ma trận đường chéo (Scipy sparse matrix).
    :param k: Số trị riêng nhỏ nhất cần tính.
    :return: Các vector riêng tương ứng (k vector).
    """
    # Chuan hoa ma tran Laplace: D^-1/2 * L * D^-1/2
    D_diag = D.diagonal().copy()  # Lay duong cheo cua D
    D_diag[D_diag < 1e-10] = 1e-10  # Tranh chia cho 0 hoac gan 0
    D_inv_sqrt = diags(1.0 / cp.sqrt(D_diag))  # Tinh D^-1/2
    L_normalized = D_inv_sqrt @ L @ D_inv_sqrt  # Chuan hoa ma tran Laplace
    
    # Khởi tạo vector ngẫu nhiên
    v0 = cp.random.rand(L.shape[0])
    v0 /= cp.linalg.norm(v0)
    
    # Áp dụng thuật toán Lanczos
    T, V = Lanczos(L_normalized, v0, m=k+5)  # Sử dụng m > k để tăng độ chính xác
    
    # Tính trị riêng và vector riêng của ma trận tam giác T
    eigvals, eigvecs_T = cp.linalg.eigh(T[:k, :k])
    
    # Chuyển đổi vector riêng về không gian gốc
    eigvecs_original = D_inv_sqrt @ (V[:k, :].T @ eigvecs_T)
    
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
    eigen_vectors = compute_eigen(L,D, k=k)  # Tinh k vector rieng
    
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
