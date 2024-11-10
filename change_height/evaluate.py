# evaluate.py
import torch
from torch.utils.data import DataLoader
# from tacnet_model import TacNet
from model import RegressionCNN
from dataset import ImageRegressionDataset
from config import TEST_CSV_FILE, TEST_IMG_DIR, DEVICE, BATCH_SIZE

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0.0
    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Test MSE: {avg_loss:.4f}")

if __name__ == "__main__":
    model = RegressionCNN()
    model.load_state_dict(torch.load("/media/namcao/NAM CAO/Ubuntu/Blackmax_regression/change_height/RegressionCNN_model.pth"))
    
    # Use the test CSV and image directory specified in config.py
    test_dataset = ImageRegressionDataset(csv_file=TEST_CSV_FILE, img_dir=TEST_IMG_DIR)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    evaluate(model, test_dataloader)
