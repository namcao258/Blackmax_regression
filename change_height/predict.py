# predict.py
import torch
# from tacnet_model import TacNet
from model import RegressionCNN
from torchvision import transforms
from PIL import Image
from config import DEVICE, MAX_X, MAX_Z, MAX_ANGLE, MAX_HEIGHT

def load_model(model_path):
    """Load the trained model."""
    model = RegressionCNN()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

def preprocess_image(image_path):
    """Load and preprocess a single image for prediction."""
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((256, 256)),                # Resize to model input size
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])            # Normalize as per training
    ])
    image = Image.open(image_path).convert("L")       # Open image as grayscale
    return transform(image).unsqueeze(0).to(DEVICE)   # Add batch dimension and send to device

def predict_single_image(model, image_path):
    """Predicts the output for a single image from the file path."""
    image = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image)
        return output.cpu().numpy()  # Convert output to numpy array for readability

def denormalize_output(output):
    """Convert normalized output back to original scale."""
    x_position = output[0] * MAX_X
    z_position = output[1] * MAX_Z
    rotation_angle = output[2] * MAX_ANGLE
    robot_height = output[3] * MAX_HEIGHT
    return x_position, z_position, rotation_angle, robot_height

if __name__ == "__main__":
    # Load the model
    model = load_model("/media/namcao/NAM CAO/Ubuntu/Blackmax_regression/change_height/RegressionCNN_model.pth")
    
    # Specify the image path directly
    image_path = "/media/namcao/NAM CAO/Ubuntu/Blackmax_regression/change_height/test_images/img_144_x15.00_z60.00_angle270.00_robot_height350.00.png"  # Replace with your actual image path
    
    # Make a prediction on the specified image
    output = predict_single_image(model, image_path)
    
    # Denormalize the output
    x_position, z_position, rotation_angle, robot_height= denormalize_output(output[0])
    
    # Display the denormalized results
    print("Predicted Output (Original Scale):")
    print(f"x_position: {x_position}")
    print(f"z_position: {z_position}")
    print(f"rotation_angle: {rotation_angle}")
    print(f"robot_height: {robot_height}")
