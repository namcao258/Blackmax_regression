# split_data.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

# Paths for the original data
original_csv_path = '/home/namcao/Desktop/Blackmax_regression/image_data_time_2.csv'
original_img_dir = '/home/namcao/Desktop/Blackmax_regression/data_bw_time_2'

# Paths for the new train and test data
train_csv_path = '/home/namcao/Desktop/Blackmax_regression/train_data.csv'
test_csv_path = '/home/namcao/Desktop/Blackmax_regression/test_data.csv'
train_img_dir = '/home/namcao/Desktop/Blackmax_regression/train_images'
test_img_dir = '/home/namcao/Desktop/Blackmax_regression/test_images'

# Create directories for train and test images if they don't exist
os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(test_img_dir, exist_ok=True)

# Load the original data and split it (70% for training, 30% for testing)
data = pd.read_csv(original_csv_path)
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

# Save the split data to new CSV files
train_data.to_csv(train_csv_path, index=False)
test_data.to_csv(test_csv_path, index=False)

# Copy images to the respective train and test directories
def copy_images(data, destination_dir):
    for _, row in data.iterrows():
        img_name = row['name']  # Assumes 'name' column in CSV has the image filename
        src_path = os.path.join(original_img_dir, img_name)
        dest_path = os.path.join(destination_dir, img_name)
        if os.path.exists(src_path):
            shutil.copy(src_path, dest_path)

# Copy images for train and test data
copy_images(train_data, train_img_dir)
copy_images(test_data, test_img_dir)

print("Data split complete. Training and testing CSV files and images are saved.")
