from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from config import Config  # Relative import for Config
from .model import ModelManager  # Relative import for ModelManager
import pandas as pd
from datasets import Dataset
from data.datasets_loader import DatasetLoader  # Relative import for DatasetLoader
import os

class TrainDataset:
    def __init__(self, data_path, preprocess_steps=None):
        self.data_path = data_path
        self.preprocess_steps = preprocess_steps if preprocess_steps is not None else []
        self.dataset = self.load_dataset()

    def ensure_file_exists(self):
        """
        Ensures that the dataset file exists. If not, creates an empty file with appropriate headers.
        """
        if not os.path.exists(self.data_path):
            with open(self.data_path, 'w') as f:
                f.write("text,answer\n")  # Placeholder headers

    def load_dataset(self):
        # Ensure the file exists
        self.ensure_file_exists()

        # Load the dataset from a CSV file
        data = pd.read_csv(self.data_path)
        self.preprocess(data)
        return Dataset.from_pandas(data)

    def preprocess(self, data):
        # Apply predefined preprocessing steps
        for step in self.preprocess_steps:
            data = step(data)
        return data


class TrainerManager:
    def __init__(self, train_dataset_path=Config.TRAIN_DATA_PATH, eval_dataset_path=Config.EVAL_DATA_PATH):
        self.model_manager = ModelManager()

        # Load and preprocess datasets
        self.train_dataset = TrainDataset(train_dataset_path).dataset
        self.eval_dataset = TrainDataset(eval_dataset_path).dataset

        self.trainer = None

    def tokenize_function(self, data):
        """
        Tokenizes the text column of the dataset.
        """
        return self.model_manager.tokenizer(data["text"], truncation=True, padding=True, max_length=Config.MAX_INPUT_LENGTH)

    def setup_trainer(self):
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.model_manager.tokenizer, mlm=False
        )

        training_args = TrainingArguments(
            output_dir=Config.OUTPUT_DIR,
            num_train_epochs=Config.NUM_TRAIN_EPOCHS,
            per_device_train_batch_size=Config.PER_DEVICE_TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=Config.PER_DEVICE_EVAL_BATCH_SIZE,
            warmup_steps=Config.WARMUP_STEPS,
            weight_decay=Config.WEIGHT_DECAY,
            logging_dir=Config.LOGGING_DIR,
        )

        self.trainer = Trainer(
            model=self.model_manager.get_model(),
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.model_manager.tokenizer,
            data_collator=data_collator,
        )

    def train(self):
        if self.trainer is None:
            self.setup_trainer()
        self.trainer.train()

    def save_trained_model(self, path):
        """
        Saves the fine-tuned model.
        """
        self.model_manager.save_model(path)

    def save_interaction(self, question, answer, data_path):
        """
        Saves a single Q&A interaction to the specified CSV file.
        """
        if not os.path.exists(data_path):
            with open(data_path, 'w') as f:
                f.write("text,answer\n")  # Placeholder headers
        # Append new interaction
        with open(data_path, 'a') as f:
            f.write(f'"{question}","{answer}"\n')
        print(f"Saved interaction to {data_path}")