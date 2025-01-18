from datasets import load_dataset
import pandas as pd


class DatasetLoader:
    """
    Class to handle loading and converting datasets for training and evaluation.
    """

    @staticmethod
    def load_csv_to_dataset(csv_path):
        """
        Loads a CSV file into a Hugging Face Dataset.

        Args:
            csv_path (str): Path to the CSV file.

        Returns:
            Dataset: A Hugging Face Dataset object.
        """
        try:
            # Convert CSV to pandas DataFrame and then to Hugging Face Dataset
            df = pd.read_csv(csv_path)
            return load_dataset("pandas", data_files={"data": csv_path})["data"]
        except Exception as e:
            print(f"Error loading CSV: {e}")
            raise

    @staticmethod
    def preprocess_text(dataset, tokenizer, max_length=512):
        """
        Tokenizes the text column of the dataset for model training.

        Args:
            dataset (Dataset): A Hugging Face Dataset object.
            tokenizer (PreTrainedTokenizer): Tokenizer for the target model.
            max_length (int): Maximum sequence length.

        Returns:
            Dataset: Tokenized dataset.
        """
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )

        return dataset.map(tokenize_function, batched=True)
