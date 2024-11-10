import cv2
import numpy as np
import torch
import os

# Hàm xử lý ảnh
def process_image(frame):
    # Chuyển đổi ảnh sang không gian màu HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Định nghĩa phạm vi màu đỏ trong HSV
    lower_red = np.array([0, 105, 79])
    upper_red = np.array([18, 255, 255])
    lower_red2 = np.array([165, 90, 113])
    upper_red2 = np.array([255, 255, 255])

    # Tạo các mặt nạ để phát hiện màu đỏ
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    # Tìm các đường bao của vùng màu đỏ
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Tạo ảnh đầu ra màu đen
    output = np.zeros_like(frame)

    # Vẽ các đường bao đã phát hiện lên ảnh đầu ra
    cv2.drawContours(output, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    # Chuyển ảnh đầu ra sang ảnh xám
    gray_output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

    # Áp dụng phép toán giãn để làm rõ các vùng trắng
    kernel = np.ones((5, 5), np.uint8)
    gray_output = cv2.dilate(gray_output, kernel, iterations=1)

    # Áp dụng phép toán đóng để làm đầy các lỗ hổng nhỏ trong các chấm tròn
    gray_output = cv2.morphologyEx(gray_output, cv2.MORPH_CLOSE, kernel)

    # Áp dụng lọc trung vị để giảm nhiễu lấm tấm
    gray_output = cv2.medianBlur(gray_output, 5)

    # Chuẩn hóa ảnh về khoảng [0, 1]
    gray_output = gray_output.astype(np.float32) / 255.0

    return gray_output  # Trả về ảnh xám đã xử lý

# Đường dẫn đến thư mục ảnh đầu vào và đầu ra
input_dir = "/media/namcao/NAM CAO/Ubuntu/Blackmax_regression/images_time_2"   # Thay bằng đường dẫn thư mục chứa ảnh gốc
output_dir = "data_bw_time_2_new"                     # Thư mục đầu ra

# Tạo thư mục đầu ra nếu chưa tồn tại
os.makedirs(output_dir, exist_ok=True)

# Duyệt qua tất cả các ảnh trong thư mục đầu vào
for filename in os.listdir(input_dir):
    if filename.endswith((".jpg", ".png", ".jpeg")):  # Chỉ xử lý các định dạng ảnh phổ biến
        # Đọc ảnh
        img_path = os.path.join(input_dir, filename)
        frame = cv2.imread(img_path)

        # Kiểm tra nếu ảnh không thể đọc
        if frame is None:
            print(f"Không thể đọc ảnh {filename}")
            continue

        # Xử lý ảnh
        processed_image = process_image(frame)

        # Chuyển ảnh từ khoảng [0, 1] về khoảng [0, 255] để lưu dưới dạng ảnh
        processed_image = (processed_image * 255).astype(np.uint8)

        # Lưu ảnh đã xử lý vào thư mục đầu ra
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, processed_image)

        print(f"Đã xử lý và lưu ảnh: {output_path}")

print("Hoàn thành xử lý tất cả ảnh.")
