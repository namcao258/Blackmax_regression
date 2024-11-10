import cv2
import numpy as np
import torch

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

    # Chuyển đổi ảnh thành tensor và thêm một chiều kênh
    image_tensor = torch.tensor(gray_output, dtype=torch.float32).unsqueeze(0)  # (1, H, W)

    return gray_output  # Trả về ảnh xám đã xử lý để hiển thị

# Khởi tạo camera
cap = cv2.VideoCapture(0)

while True:
    # Đọc ảnh từ camera
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc ảnh từ camera")
        break

    # Xử lý ảnh
    processed_image = process_image(frame)

    # Hiển thị ảnh đã xử lý
    cv2.imshow("Processed Image", processed_image)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng camera và đóng các cửa sổ hiển thị
cap.release()
cv2.destroyAllWindows()
