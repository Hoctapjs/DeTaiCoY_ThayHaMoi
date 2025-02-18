import cupy as cp  # Thay thế NumPy bằng CuPy
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from cupyx.scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from skimage import io, color
import cupyx.scipy.sparse as sp
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import time
import logging
import os

def kiemThuChayNhieuLan(i, name, folder_path):
    # Kiểm tra xem thư mục có tồn tại không
    if not os.path.isdir(folder_path):
        print(f"❌ Thư mục {folder_path} không tồn tại!")
        return
    
    # Lấy danh sách tất cả ảnh trong thư mục
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    if not image_files:
        print(f"❌ Không tìm thấy file ảnh nào trong {folder_path}!")
        return

    for idx, file_name in enumerate(image_files, start=1):
        image_path = os.path.join(folder_path, file_name)

        # Tạo file log riêng cho từng lần chạy
        log_file = f"{name}_{i}_{idx}.txt"
        save_image_name = f"{name}_{i}_{idx}.png"

        
        # Cấu hình logging
        logging.basicConfig(filename=log_file, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

        print(f"📷 Đang xử lý ảnh {idx}: {image_path}")
        
        # Gọi hàm xử lý ảnh
        normalized_cuts(i, file_name, image_path, save_image_name)

def compute_weight_matrix(image, sigma_i=0.1, sigma_x=10):
    h, w, c = image.shape

    # Toa do (x, y)
    coords = cp.array(cp.meshgrid(cp.arange(h), cp.arange(w))).reshape(2, -1).T

    # Dac trung mau
    features = cp.array(image).reshape(-1, c)

    
    # Tinh ma tran trong so bang vector hoa
    # tính thời gian - 1 hình - 10 lần 50 70 90 (tối thiểu) - 10 ảnh
    # start
    W_color = cp.array(rbf_kernel(features.get(), gamma=1 / (2 * sigma_i**2)))
    W_space = cp.array(rbf_kernel(coords.get(), gamma=1 / (2 * sigma_x**2)))
    # end
    W = W_color * W_space
    # end đo thời gian W
    return W

# 2. Tinh ma tran Laplace
def compute_laplacian(W):
    D = cp.diag(W.sum(axis=1))  # Ma trận đường chéo
    L = D - W
    
    return L, D

""" # 3. Giai bai toan tri rieng
def compute_eigen(L, D, k=2):
    # Chuyen du lieu ve CPU vi eigsh chua ho tro GPU
    L_cpu, D_cpu = L.get(), D.get()
    vals, vecs = eigsh(L_cpu, k=k, M=D_cpu, which='SM')  # 'SM' tim tri rieng nho nhat
    return cp.array(vecs)  # Tra ve k vector rieng (chuyen ve GPU) """
# 3. Giai bai toan tri rieng
def compute_eigen(L, k=2):
    # Tìm các trị riêng nhỏ nhất (Smallest Magnitude)
    eigvals, eigvecs = eigsh(L, k=k, which='SA')  
    return eigvecs

# 4. Gan nhan cho tung diem anh dua tren vector rieng
def assign_labels(eigen_vectors, k):
    # Chuyen du lieu ve CPU de dung K-Means
    eigen_vectors_cpu = eigen_vectors.get()

    kmeans = KMeans(n_clusters=k, random_state=0).fit(eigen_vectors_cpu)
    labels = kmeans.labels_
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
def normalized_cuts(lan, imagename, image_path, output_path):
    
    # Tinh toan tren GPU
    start_gpu = time.time()
    logging.info(f"file name: {imagename}")
    logging.info(f"Lan thu: {lan}")
    # Doc anh va chuan hoa
    image = io.imread(image_path)
    if image.ndim == 2:  # Neu la anh xam, chuyen thanh RGB
        image = color.gray2rgb(image)
    elif image.shape[2] == 4:  # Neu la anh RGBA, loai bo kenh alpha
        image = image[:, :, :3]
    image = image / 255.0  # Chuan hoa ve [0, 1]
    
    k=3

    # Tinh toan Ncuts
    start_cpu_coo = time.time()
    W = compute_weight_matrix(image)
    end_cpu_coo = time.time()
    
    L, D = compute_laplacian(W)
    
    eigen_vectors = compute_eigen(L, k=k)  # Tinh k vector rieng
    
    labels = assign_labels(eigen_vectors, k)  # Gan nhan cho moi diem anh
    

    cp.cuda.Stream.null.synchronize()  # Dong bo hoa de dam bao GPU hoan thanh tinh toan
    end_gpu = time.time()
    logging.info(f"Thoi gian: {end_gpu - start_gpu} giay")
    logging.info(f"Thoi gian W: {end_cpu_coo - start_cpu_coo} giay")
    
    # ✅ Giải phóng bộ nhớ GPU sau khi sử dụng
    del W, L, D, eigen_vectors
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    cp.cuda.Device(0).synchronize()  # Đảm bảo giải phóng hoàn toàn

# 7. Mo file chon anh tu hop thoai
# def open_file_dialog():
#     # Tao cua so an cho tkinter
#     root = Tk()
#     root.withdraw()  # An cua so chinh
    
#     # Mo hop thoai chon file anh
#     file_path = askopenfilename(title="Chon anh", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
#     return file_path   

""" # 8. Chay thu nghiem
if __name__ == "__main__":
    # Mo hop thoai chon anh
    image_path = open_file_dialog()
    if image_path:
        logging.info(f"Da chon anh: {image_path}")
        normalized_cuts(image_path, k=3)  # Phan vung thanh 3 nhom
    else:
        logging.info("Khong co anh nao duoc chon.") """
