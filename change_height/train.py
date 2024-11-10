# train.py
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
# from tacnet_model import TacNet
from model import RegressionCNN
from dataset import ImageRegressionDataset
from config import TRAIN_CSV_FILE, TRAIN_IMG_DIR, DEVICE, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS

def main():
    model = RegressionCNN()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Use the training CSV and image directory specified in config.py
    dataset = ImageRegressionDataset(csv_file=TRAIN_CSV_FILE, img_dir=TRAIN_IMG_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        model.train()
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", unit="batch") as pbar:
            for images, labels in dataloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix(loss=running_loss / (pbar.n + 1))
                pbar.update(1)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {running_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), "RegressionCNN_model.pth")
    print("Model saved to RegressionCNN_model.pth")

if __name__ == "__main__":
    main()
