import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse.linalg import eigsh, LinearOperator
from scipy.sparse import diags
from sklearn.cluster import KMeans
from skimage import io, color
import time
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from scipy.sparse import coo_matrix #chuyển sang ma trận coo
from scipy.sparse import isspmatrix, diags
import logging


# logging.basicConfig(level=logging.INFO)  

def kiemThuChayNhieuLan(i, name):
        temp_chuoi = f"{name}{i}"
        temp_chuoi = temp_chuoi + '.txt'
        logging.basicConfig(filename = temp_chuoi, level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
        # Duong dan toi anh cua ban
        image_path = "apple3_60x60.jpg"  # Thay bang duong dan anh cua ban
        """ image_path = "apple4_98x100.jpg"  # Thay bang duong dan anh cua ban """
        normalized_cuts(image_path, k=3)
        


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
    D_diag = W_sparse.sum(axis=1).A.flatten() if hasattr(W_sparse, 'toarray') else W_sparse.sum(axis=1)
    D = np.diag(D_diag)  # Ma trận đường chéo
    L = D - W_sparse.toarray() if hasattr(W_sparse, 'toarray') else D -W_sparse  # Đảm bảo W là dạng mảng NumPy


    # Tạo ma trận đường chéo từ tổng các hàng
    # D_diag = W_sparse.sum(axis=1).A.flatten() 
    # D = np.diag(D_diag)  # Ma trận đường chéo
    # L = D - W_sparse # L = D - W

    logging.info("Kich thuoc ma tran duong cheo (D): %s", D.shape)
    logging.info("Mau cua D (9 phan tu dau):\n%s", D_diag[:9])  # In phần tử trên đường chéo
    logging.info("Kich thuoc ma tran Laplace (L): %s", L.shape)
    logging.info("Mau cua L (9x9 phan tu dau):\n%s", L[:9, :9])

    return L, D

# def compute_laplacian(W_sparse):
#     D_diag = np.array(W_sparse.sum(axis=1)).flatten()
#     D_inv_sqrt = diags(1.0 / np.sqrt(D_diag + 1e-10))  # Tránh chia cho 0
#     L_normalized = D_inv_sqrt @ (diags(D_diag) - W_sparse) @ D_inv_sqrt
#     return L_normalized, D_inv_sqrt

# def compute_eigen(L_normalized, k=2):
#     eigvals, eigvecs = eigsh(L_normalized, k=k, which='SM')
#     return eigvecs



# 3. Giai bai toan tri rieng

#Version 3 - không dùng eigsh mà code lanzcos thuần - song vẫn dùng eig
def Lanczos(A, v, m):
    """
    Thuật toán Lanczos để xấp xỉ trị riêng và vector riêng.
    : A: Ma trận cần tính (numpy 2D array).
    : v: Vector khởi tạo.
    : m: Số bước lặp Lanczos.
    :return: Ma trận tam giác T và ma trận trực giao V.
    """
    n = len(v) # Đây là số phần tử trong vector v (số chiều của ma trận A)
    V = np.zeros((m, n)) # đây là một ma trận mxn lưu trữ các vector trực giao (là 2 vector có tích vô hướng = 0), mỗi hàng là một bước đã đi qua, np.zeros nghĩa là ban đầu tất cả các bước đi (hay các phần tử của ma trận) đều là 0, chưa đi bước nào
    T = np.zeros((m, m)) # đây là ma trận tam giác T
    V[0, :] = v / np.linalg.norm(v) # np.linalg.norm(v) là để tính chuẩn (độ dài) của vector = căn(v1^2 + v2^2 + ...)
    # => V[0, :] = v / np.linalg.norm(v) là để chuẩn hóa vector v đầu vào thành vector đơn vị 
    
    # Đoạn này là để làm cho w trực giao với V0 thôi
    # vd: để làm cho 2 vector a và b trực giao với nhau
    # 1. tính tích vô hướng của a và b (alpha)
    # 2. cập nhật vector a lại 
    #   a = a - alpha * b (b ở đây là V[0, :] = v / căn(v) )


    w = A @ V[0, :] # tính vector w bằng cách nhân A với vector đầu tiên của V - hiểu nôm na là w sẽ cho ta biết các mà ma trận A tương tác với vector khởi tạo v
    alpha = np.dot(w, V[0, :]) # .dot là tính tích vô hướng của 2 vector a và b (trong case này là w và vector đầu tiên của V), hệ số alpha là để đo mức độ song song giữa w và V0
    w = w - alpha * V[0, :]
    # alpha * V[0, :] tạo ra một vector có hướng song song với 
    # V[0,:] mà có độ dài tương ứng.
    # sau khi trừ xong thì nò sẽ loại bỏ phần song song ra khỏi w

    
    T[0, 0] = alpha # Gán giá trị alpha vào phần tử đầu tiên của T
    
    for j in range(1, m):
        beta = np.linalg.norm(w)
        if beta < 1e-10:
            break
        V[j, :] = w / beta
        w = A @ V[j, :]
        alpha = np.dot(w, V[j, :])
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
    D_inv_sqrt = diags(1.0 / np.sqrt(D_diag))  # Tinh D^-1/2
    L_normalized = D_inv_sqrt @ L @ D_inv_sqrt  # Chuan hoa ma tran Laplace
    
    # Khởi tạo vector ngẫu nhiên
    v0 = np.random.rand(L.shape[0])
    v0 /= np.linalg.norm(v0)
    
    # Áp dụng thuật toán Lanczos
    T, V = Lanczos(L_normalized, v0, m=k+5)  # Sử dụng m > k để tăng độ chính xác
    
    # Tính trị riêng và vector riêng của ma trận tam giác T
    eigvals, eigvecs_T = np.linalg.eig(T[:k, :k])
    
    # Chuyển đổi vector riêng về không gian gốc
    eigvecs_original = D_inv_sqrt @ (V[:k, :].T @ eigvecs_T)
    
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
    
    # Tinh toan Ncuts
    start_cpu_coo = time.time()
    logging.info("Bat dau tinh ma tran trong so...")
    W = compute_weight_matrix(image)
    end_cpu_coo = time.time()
    
    logging.info("Tinh ma tran Laplace...")
    L, D = compute_laplacian(W)
    
    logging.info("Tinh vector rieng...")
    eigen_vectors = compute_eigen(L, D, k=k)  # Tinh k vector rieng
    # eigen_vectors = compute_eigen(W, k=k)  # Tinh k vector rieng
    
    logging.info("Gan nhan cho cac diem anh...")
    labels = assign_labels(eigen_vectors, k)  # Gan nhan cho moi diem anh
    
    logging.info("Hien thi ket qua...")

    end_cpu = time.time()
    logging.info(f"Thoi gian: {end_cpu - start_cpu} giay")
    logging.info(f"Thoi gian COO: {end_cpu_coo - start_cpu_coo} giay")

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

