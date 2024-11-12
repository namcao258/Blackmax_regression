# checkpoint
import cv2
import numpy as np
import torch
from model import RegressionCNN  # Import mô hình dự đoán
from torchvision import transforms
from PIL import Image
from config import DEVICE, MAX_X, MAX_Z, MAX_ANGLE, MAX_HEIGHT
# import test_display

def process_image(frame):
    # Chuyển đổi ảnh sang không gian màu HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 105, 79])
    upper_red = np.array([18, 255, 255])
    lower_red2 = np.array([165, 90, 113])
    upper_red2 = np.array([255, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = np.zeros_like(frame)
    cv2.drawContours(output, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    gray_output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.uint8)
    gray_output = cv2.dilate(gray_output, kernel, iterations=1)
    gray_output = cv2.morphologyEx(gray_output, cv2.MORPH_CLOSE, kernel)
    gray_output = cv2.medianBlur(gray_output, 5)
    gray_output = gray_output.astype(np.float32) / 255.0
    return gray_output

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

def load_model(model_path):
    model = RegressionCNN().to(DEVICE) 
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

def predict_single_image(model, image):
    image_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(image_tensor)
        return output.cpu().numpy() 

def denormalize_output(output):
    x_position = output[0] * MAX_X
    z_position = output[1] * MAX_Z
    rotation_angle = output[2] * MAX_ANGLE
    robot_height = output[3] * MAX_HEIGHT
    return x_position, z_position, rotation_angle, robot_height

cap = cv2.VideoCapture(0)

model = load_model("/media/namcao/NAM CAO/Ubuntu/Blackmax_regression/change_height/RegressionCNN_model.pth")
file_path = "/media/namcao/NAM CAO/Ubuntu/Blackmax_regression/change_height/skin_5.vtk"
plotter = None 
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can not read camera")
        break

    processed_image = process_image(frame)
    pil_image = Image.fromarray((processed_image * 255).astype(np.uint8))

    # Dự đoán các thông số từ ảnh đã xử lý
    output = predict_single_image(model, pil_image)
    x_position, z_position, rotation_angle, robot_height = denormalize_output(output[0])

    print("Predicted Output (Original Scale):")
    print(f"x_position: {x_position}")
    print(f"z_position: {z_position}")
    print(f"rotation_angle: {rotation_angle}")
    print(f"robot_height: {robot_height}")
    cv2.imshow("Processed Image", processed_image)
    # if(z_position>25):
        # test_display.display_model_with_camera(file_path,robot_height,z_position,rotation_angle,x_position, processed_image)
        # test_display.display_predict(x_position,z_position, rotation_angle, robot_height)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
plotter.close()
