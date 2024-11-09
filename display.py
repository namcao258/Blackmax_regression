import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import torch
from tacnet_model import TacNet
from config import DEVICE, MAX_X, MAX_Z, MAX_ANGLE, MAX_HEIGHT

# Cylinder and display parameters
radius = 110
height = 200
theta_resolution = 360
z_resolution = 301
contact_depth_max = 45

def load_model(model_path):
    """Load the trained model."""
    model = TacNet().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

def process_image(frame):
    """Preprocess the camera frame to detect red regions and prepare for prediction."""
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

    # Resize to 256x256
    gray_output = cv2.resize(gray_output, (256, 256), interpolation=cv2.INTER_AREA)

    # Normalize the image to range [0, 1]
    gray_output = gray_output.astype(np.float32) / 255.0

    # Convert the image to a tensor and add batch and channel dimensions
    image_tensor = torch.tensor(gray_output, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 256, 256)

    return image_tensor.to(DEVICE)

def predict_frame(model, frame):
    """Predicts the output for a single preprocessed frame from the camera."""
    image_tensor = process_image(frame)  # Use the custom preprocessing
    with torch.no_grad():
        output = model(image_tensor).cpu().numpy()[0]
    x_position, z_position, rotation_angle, robot_height = output
    return x_position * MAX_X, z_position * MAX_Z, rotation_angle * MAX_ANGLE, robot_height * MAX_HEIGHT

def update_display(ax, x_position, z_position, rotation_angle, robot_height, text_boxes):
    """Update the 3D plot and text boxes with new predicted contact points and depth."""
    theta_index = int((rotation_angle / 360) * theta_resolution) % theta_resolution
    z_index = int((z_position / height) * z_resolution) % z_resolution

    theta = np.linspace(0, 2 * np.pi, theta_resolution)
    z_vals = np.linspace(0, height, z_resolution)
    theta_grid, z_grid = np.meshgrid(theta, z_vals)
    x_grid = radius * np.cos(theta_grid)
    y_grid = radius * np.sin(theta_grid)

    # Clear previous plot
    ax.cla()
    ax.plot_surface(x_grid, y_grid, z_grid, color='cyan', alpha=0.4, edgecolor='gray')
    ax.plot([0, 0], [0, 0], [0, height], color='red', linewidth=2)

    # Calculate Cartesian coordinates of predicted point
    node_x = x_grid[z_index, theta_index]
    node_y = y_grid[z_index, theta_index]
    node_z = z_grid[z_index, theta_index]

    # Normalize robot_height for display as contact depth color intensity
    norm = mcolors.Normalize(vmin=0, vmax=contact_depth_max)
    cmap = plt.get_cmap('Reds')
    node_color = cmap(norm(robot_height))

    # Highlight the predicted node and add an arrow for contact depth
    ax.scatter(node_x, node_y, node_z + 5, color=node_color, s=300, edgecolor='black', linewidth=2.5)
    arrow_length = robot_height / contact_depth_max * radius
    arrow_dx = -node_x * (arrow_length / radius)
    arrow_dy = -node_y * (arrow_length / radius)
    ax.quiver(node_x, node_y, node_z + 5, arrow_dx, arrow_dy, 0, color='black', linewidth=1.5)

    # Set plot labels and title
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('3D Cylinder with Predicted Contact Depth')
    ax.set_box_aspect([1, 1, height / radius])

    # Update text boxes with prediction results
    text_boxes['x_position'].set_text(f"contact_depth: {x_position:.2f}")
    text_boxes['z_position'].set_text(f"z_position: {z_position:.2f}")
    text_boxes['rotation_angle'].set_text(f"rotation_angle: {rotation_angle:.2f}")
    text_boxes['robot_height'].set_text(f"robot_height: {robot_height:.2f}")



def predict_and_display_camera(model):
    """Captures frames from the camera, predicts, and displays in real-time."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Text boxes for displaying predictions
    text_boxes = {
        'x_position': fig.text(0.05, 0.95, "contact_depth:: ", fontsize=12, color="blue"),
        'z_position': fig.text(0.05, 0.90, "z_position: ", fontsize=12, color="blue"),
        'rotation_angle': fig.text(0.05, 0.85, "rotation_angle: ", fontsize=12, color="blue"),
        'robot_height': fig.text(0.05, 0.80, "robot_height: ", fontsize=12, color="blue"),
    }

    # Open the camera
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Predict contact points from frame
            x, z, angle, contact_depth = predict_frame(model, frame)

            # Update display with predictions
            update_display(ax, x, z, angle, contact_depth, text_boxes)
            plt.pause(0.01)

            # Display the camera feed in a separate window
            cv2.imshow("Camera Feed", frame)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        plt.close(fig)  # Ensure the matplotlib figure closes when the loop ends

if __name__ == "__main__":
    # Load the model
    model = load_model("tacnet_model.pth")
    
    # Predict and display real-time from the camera
    predict_and_display_camera(model)