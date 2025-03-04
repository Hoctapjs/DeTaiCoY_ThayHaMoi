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
import pandas as pd
from cupyx.scipy.sparse import diags

def kiemThuChayNhieuLan(i, name, folder_path, output_excel="results.xlsx"):
    # Kiểm tra thư mục
    if not os.path.isdir(folder_path):
        print(f"❌ Thư mục {folder_path} không tồn tại!")
        return
    
    # Lấy danh sách file ảnh
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    if not image_files:
        print(f"❌ Không tìm thấy file ảnh nào trong {folder_path}!")
        return

    results = []  # Danh sách lưu kết quả

    for idx, file_name in enumerate(image_files, start=1):
        image_path = os.path.join(folder_path, file_name)
        print(f"📷 Đang xử lý ảnh {idx}: {image_path}")
        
        # Xử lý ảnh
        _, wf_time, wc_time   = normalized_cuts(i, file_name, image_path, output_excel)  # Bỏ time_w vì không cần
        
        # Lưu kết quả vào danh sách
        results.append([i, idx, file_name, wf_time, wc_time])

    # Ghi kết quả vào file Excel
    df = pd.DataFrame(results, columns=["Lần chạy", "Ảnh số", "Tên ảnh", "Thời gian W đặc trưng (s)", "Thời gian W tọa độ (s)"])


    # Tạo tên file kết quả theo format chuẩn
    output_excel = f"result_{name}_{i}.xlsx"
    
    df.to_excel(output_excel, index=False, engine='openpyxl')
    print(f"✅ Kết quả đã lưu vào {output_excel}")

def compute_weight_matrix(image, sigma_i=0.1, sigma_x=10):
    h, w, c = image.shape

    coords = cp.array(cp.meshgrid(cp.arange(h), cp.arange(w))).reshape(2, -1).T

    features = cp.array(image).reshape(-1, c)

    
    # Tinh ma tran trong so bang vector hoa
    # tính thời gian - 1 hình - 10 lần 50 70 90 (tối thiểu) - 10 ảnh
    # start

    # tính thời gian w đặc trưng
    start_features = time.time()
    W_color = cp.array(rbf_kernel(features.get(), gamma=1 / (2 * sigma_i**2)))
    end_features = time.time()

    W_features_time =  end_features - start_features

    # tính thời gian w tọa độ
    start_coords = time.time()
    W_space = cp.array(rbf_kernel(coords.get(), gamma=1 / (2 * sigma_x**2)))
    end_coords = time.time()

    W_coords_time =  end_coords - start_coords

    
    
    # end
    W = W_color * W_space
    # end đo thời gian W
    return W, W_features_time, W_coords_time

# 2. Tinh ma tran Laplace
def compute_laplacian(W):
    D = cp.diag(W.sum(axis=1))  # Ma trận đường chéo
    L = D - W
    
    return L, D

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
    W, W_f, W_c = compute_weight_matrix(image)
    end_cpu_coo = time.time()
    
    L, D = compute_laplacian(W)
    
    eigen_vectors = compute_eigen(L, D, k=k)  # Tinh k vector rieng
    
    labels = assign_labels(eigen_vectors, k)  # Gan nhan cho moi diem anh
    

    cp.cuda.Stream.null.synchronize()  # Dong bo hoa de dam bao GPU hoan thanh tinh toan
    end_gpu = time.time()
    logging.info(f"Thoi gian: {end_gpu - start_gpu} giay")
    logging.info(f"Thoi gian W: {end_cpu_coo - start_cpu_coo} giay")
    
    # del W, L, D, eigen_vectors
    # cp.get_default_memory_pool().free_all_blocks()
    # cp.get_default_pinned_memory_pool().free_all_blocks()
    # cp.cuda.Device(0).synchronize()

    return (end_cpu_coo - start_cpu_coo), W_f, W_c

    # # ✅ Giải phóng bộ nhớ GPU sau khi sử dụng
    # del W, L, D, eigen_vectors
    # cp.get_default_memory_pool().free_all_blocks()
    # cp.get_default_pinned_memory_pool().free_all_blocks()
    # cp.cuda.Device(0).synchronize()  # Đảm bảo giải phóng hoàn toàn

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
