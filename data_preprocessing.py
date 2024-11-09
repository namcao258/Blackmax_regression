import cv2
import torch
import numpy as np
from example import UNetModel  # Import UNetModel from example.py

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# Image preprocessing function based on the provided process_image
def process_image(frame):
    # Convert the frame to HSV (Hue, Saturation, Value) color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range for red color in HSV
    lower_red = np.array([0, 105, 79])
    upper_red = np.array([18, 255, 255])
    lower_red2 = np.array([165, 90, 113])
    upper_red2 = np.array([255, 255, 255])

    # Create masks to detect red color
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    # Find contours of the red areas
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an output image that is black everywhere
    output = np.zeros_like(frame)

    # Draw the detected contours in white on the output image
    cv2.drawContours(output, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    # Convert the output image to grayscale
    gray_output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

    # Apply a morphological operation to remove small white areas (less than 2 pixels)
    kernel = np.ones((2, 2), np.uint8)
    gray_output = cv2.morphologyEx(gray_output, cv2.MORPH_OPEN, kernel)

    # Normalize the image to range [0, 1]
    gray_output = gray_output.astype(np.float32) / 255.0

    # Convert the image to a tensor and add a channel dimension
    image_tensor = torch.tensor(gray_output, dtype=torch.float32).unsqueeze(0)  # (1, H, W)

    return image_tensor

# Function to load the trained UNet model
def load_model(model_path, output_size):
    model = UNetModel(output_size=output_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set to evaluation mode
    return model

# Function to predict on a single image (real-time frame)
def predict_image(model, image_tensor):
    image_tensor = image_tensor.to(device).unsqueeze(0)  # Add batch dimension (N=1)
    with torch.no_grad():  # Disable gradient computation for inference
        output = model(image_tensor)
    return output.cpu().numpy()  # Return the output as a numpy array

# Main function for real-time prediction using a camera
def main():
    # Initialize the camera
    cap = cv2.VideoCapture(0)  # 0 is usually the default camera

    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    model_path = './unet_model_best.pth'  # Path to the trained UNet model
    output_size = 4  # Adjust based on your model's output

    # Load the trained UNet model
    model = load_model(model_path, output_size)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame from camera.")
            break

        # Process the frame
        processed_image = process_image(frame)

        # Predict using the model
        prediction = predict_image(model, processed_image)
        print(f"Prediction: {prediction}")

        # Display the original frame and the processed image
        cv2.imshow('Camera Feed', frame)
        cv2.imshow('Processed Image', processed_image.squeeze().cpu().numpy())

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
