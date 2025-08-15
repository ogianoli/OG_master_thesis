# --- Hyperparameters --- #
NUM_CLASSES = 3
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 8e-6
WEIGHT_DECAY = 1e-5

# --- Other Config Data
INPUT_LENGTH = 100  # Resize/Pad all functions to same length
TRAIN_CSV_PATH = "master_thesis/my_functions/CNN/data/functions_train.csv"
TEST_CSV_PATH = "master_thesis/my_functions/CNN/data/functions_test.csv"
VALIDATION_CSV_PATH = "master_thesis/my_functions/CNN/data/functions_validation.csv"
RESAMPLING = False  # If True, resample the dataset to balance classes

MODEL_PATH = "unet1d_model_2.pth"
SOURCE_CSV_PATH = "master_thesis/my_functions/CNN/data/functions.csv"
TRAIN_TEST_CSV_PATH = "master_thesis/my_functions/CNN/data/functions_train_test.csv"  # Remaining data (used for train/test)

''' --- Training configurations for different runs ---
run2:   LEARNING_RATE = 1e-3
        weight_decay= not exoisting
        campaign accuracy: 99.6%
        comment: overfitting, good looking training loss but not test loss

run4:   LEARNING_RATE = 1e-4
        weight_decay=1e-5
        campaign accuracy: 99.6%
        comment: good results and better looking test loss

run5:   LEARNING_RATE = 1e-4
        weight_decay= not existing
        campaign accuracy: 98.4%
        comment: less accurate, but missclassified cases make sense bc look like bates...

run6:   LEARNING_RATE = 1e-5
        weight_decay= 1e-5
        campaign accuracy: 99.2%
        comment: loss should decay slower but be more accurate. (slower learning but more accurate)

run7:   LEARNING_RATE = 1e-5
        weight_decay= 1e-4
        campaign accuracy: 99.2%
        comment: less good.

run8:   LEARNING_RATE = 1e-6
        weight_decay= 1e-5
        campaign accuracy: 98.0%
        comment: lr too tiny...

run9:   LEARNING_RATE = 8e-6
        weight_decay= 1e-5
        campaign accuracy: 99.2%
        comment: same as 6 almost, little bit better

run10:  LEARNING_RATE = 8e-6
        weight_decay= 1e-5
        campaign accuracy: 99.2%
        comment: Same as 9, but with RESAMPLING = True
'''