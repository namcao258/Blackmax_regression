# config.py
import torch

# Paths for training and test data
TRAIN_CSV_FILE = '/media/namcao/NAM CAO/Ubuntu/Blackmax_regression/change_height/train_data.csv'
TRAIN_IMG_DIR = '/media/namcao/NAM CAO/Ubuntu/Blackmax_regression/change_height/train_images'

TEST_CSV_FILE = '/media/namcao/NAM CAO/Ubuntu/Blackmax_regression/change_height/test_data.csv'
TEST_IMG_DIR = '/media/namcao/NAM CAO/Ubuntu/Blackmax_regression/change_height/test_images'

# Normalization values
MAX_X = 45.0
MAX_Z = 300
MAX_ANGLE = 360.0
MAX_HEIGHT = 350

# Model training parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
