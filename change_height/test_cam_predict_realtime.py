import cv2
import numpy as np
import torch
from model import RegressionCNN  # Import mô hình dự đoán
from torchvision import transforms
from PIL import Image
from config import DEVICE, MAX_X, MAX_Z, MAX_ANGLE, MAX_HEIGHT

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

# Hàm xử lý ảnh cho dự đoán
def preprocess_image(image):
    """Chuyển ảnh thành tensor và chuẩn hóa"""
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Chuyển ảnh sang grayscale
        transforms.Resize((256, 256)),                # Resize ảnh
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])            # Chuẩn hóa
    ])
    return transform(image).unsqueeze(0).to(DEVICE)   # Thêm chiều batch và đưa lên device

def load_model(model_path):
    """Tải mô hình đã huấn luyện"""
    model = RegressionCNN()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

def predict_single_image(model, image):
    """Dự đoán đầu ra cho một ảnh duy nhất"""
    image_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(image_tensor)
        return output.cpu().numpy()  # Chuyển output thành numpy array để dễ đọc

def denormalize_output(output):
    """Chuyển kết quả đã chuẩn hóa về lại thang đo ban đầu"""
    x_position = output[0] * MAX_X
    z_position = output[1] * MAX_Z
    rotation_angle = output[2] * MAX_ANGLE
    robot_height = output[3] * MAX_HEIGHT
    return x_position, z_position, rotation_angle, robot_height

# Khởi tạo camera
cap = cv2.VideoCapture(0)

# Tải mô hình đã huấn luyện
model = load_model("/media/namcao/NAM CAO/Ubuntu/Blackmax_regression/change_height/RegressionCNN_model.pth")

while True:
    # Đọc ảnh từ camera
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc ảnh từ camera")
        break

    # Xử lý ảnh
    processed_image = process_image(frame)

    # Chuyển ảnh xử lý thành ảnh PIL để dự đoán
    pil_image = Image.fromarray((processed_image * 255).astype(np.uint8))

    # Dự đoán các thông số từ ảnh đã xử lý
    output = predict_single_image(model, pil_image)

    # Đưa kết quả về thang đo ban đầu
    x_position, z_position, rotation_angle, robot_height = denormalize_output(output[0])

    # Hiển thị kết quả dự đoán
    print("Predicted Output (Original Scale):")
    print(f"x_position: {x_position}")
    print(f"z_position: {z_position}")
    print(f"rotation_angle: {rotation_angle}")
    print(f"robot_height: {robot_height}")

    # Hiển thị ảnh đã xử lý
    cv2.imshow("Processed Image", processed_image)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng camera và đóng các cửa sổ hiển thị
cap.release()
cv2.destroyAllWindows()
