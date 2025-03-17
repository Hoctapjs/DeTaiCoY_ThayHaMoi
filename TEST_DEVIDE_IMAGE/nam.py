import cv2
import numpy as np
import scipy.io as sio

# Đọc ảnh
image_path = "2092.jpg"
image = cv2.imread(image_path)

# Kích thước ảnh
height, width, _ = image.shape

# Cắt ảnh theo chiều dọc (hoặc thay width bằng height để cắt ngang)
left_half = image[:, :width // 2]
right_half = image[:, width // 2:]

# Lưu hai phần ảnh
cv2.imwrite("2092_left.jpg", left_half)
cv2.imwrite("2092_right.jpg", right_half)

print("Ảnh đã được cắt thành hai phần và lưu lại.")


# Đọc dữ liệu ground truth
mat_path = "2092.mat"
mat_data = sio.loadmat(mat_path)

# Giả sử groundtruth lưu trong biến 'groundTruth'
ground_truth = mat_data['groundTruth']

# Kiểm tra kích thước của ground truth
gt_height, gt_width = ground_truth.shape[:2]

# Cắt thành hai phần
left_gt = ground_truth[:, :gt_width // 2]
right_gt = ground_truth[:, gt_width // 2:]

# Lưu hai phần ground truth
sio.savemat("2092_left.mat", {'groundTruth': left_gt})
sio.savemat("2092_right.mat", {'groundTruth': right_gt})

print("Ground truth đã được cắt thành hai phần và lưu lại.")
