import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from skimage import io, color
import time
import logging

def kiemThuChayNhieuLan(i, name):
        temp_chuoi = f"{name}{i}"
        temp_chuoi = temp_chuoi + '.txt'
        logging.basicConfig(filename = temp_chuoi, level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
        # Duong dan toi anh cua ban
        """ image_path = "apple3_60x60.jpg"  # Thay bang duong dan anh cua ban """
        image_path = "apple4_98x100.jpg"  # Thay bang duong dan anh cua ban
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
    
    
    return W

# 2. Tinh ma tran Laplace
def compute_laplacian(W):
    D = np.diag(W.sum(axis=1))  # Ma tran duong cheo
    L = D - W
    
    logging.info("Kich thuoc ma tran duong cheo (D): %s", D.shape)
    logging.info("Mau cua D (9x9 phan tu dau):\n%s", D[:9, :9])
    logging.info("Kich thuoc ma tran Laplace (L): %s", L.shape)
    logging.info("Mau cua L (9x9 phan tu dau):\n%s", L[:9, :9])

    
    return L, D

# 3. Giai bai toan tri rieng
def compute_eigen(L, D, k):
    # Giai bai toan tri rieng tong quat
    vals, vecs = eigsh(L, k=k, M=D, which='SM')  # 'SM' tim tri rieng nho nhat
    
    logging.info(f"Tri rieng (Eigenvalues): {vals}")
    logging.info(f"Kich thuoc vector rieng: {vecs.shape}")
    logging.info(f"Mau cua vector rieng (9 hang dau):\n{vecs[:9, :]}")

    return vecs  # Tra ve k vector rieng

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
    
    logging.info("Gan nhan cho cac diem anh...")
    labels = assign_labels(eigen_vectors, k)  # Gan nhan cho moi diem anh
    
    logging.info("Hien thi ket qua...")

    end_cpu = time.time()
    logging.info(f"Thoi gian: {end_cpu - start_cpu} giay")
    logging.info(f"Thoi gian COO: {end_cpu_coo - start_cpu_coo} giay")

    display_segmentation(image, labels, k)

# 7. Chay thu nghiem
""" if __name__ == "__main__": """

