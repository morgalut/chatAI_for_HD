# C:\Users\Mor\Desktop\conversational_ai_project\src\config.py

class Config:
    # Model configuration
    MODEL_NAME = 'gpt2'  # Example model, change as needed
    MODEL_PATH = 'models/path_to_saved_model'  # Path to save/load the model

    # Dataset configuration
    TRAIN_DATA_PATH = 'path_to_train_data.csv'
    EVAL_DATA_PATH = 'path_to_eval_data.csv'

    # Training configuration
    OUTPUT_DIR = './results'  # Where to store training outputs
    NUM_TRAIN_EPOCHS = 3
    PER_DEVICE_TRAIN_BATCH_SIZE = 16
    PER_DEVICE_EVAL_BATCH_SIZE = 64
    WARMUP_STEPS = 500
    WEIGHT_DECAY = 0.01
    LOGGING_DIR = './logs'  # Directory for training logs

    # Tokenizer and input preparation
    MAX_INPUT_LENGTH = 512  # Maximum sequence length for model inputs
