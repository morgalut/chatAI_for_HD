import os
import logging
from config import Config
from training.model import ModelManager
from training.train import TrainerManager
from data.datasets_loader import DatasetLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure necessary directories and files exist
def ensure_directories_and_files():
    # Create model path directory
    if not os.path.exists(Config.MODEL_PATH):
        os.makedirs(Config.MODEL_PATH)
        logger.info(f"Created model directory: {Config.MODEL_PATH}")

    # Create output directory
    if not os.path.exists(Config.OUTPUT_DIR):
        os.makedirs(Config.OUTPUT_DIR)
        logger.info(f"Created output directory: {Config.OUTPUT_DIR}")

    # Create training and evaluation dataset files if they don't exist
    if not os.path.exists(Config.TRAIN_DATA_PATH):
        with open(Config.TRAIN_DATA_PATH, 'w') as f:
            f.write("text,answer\n")  # Placeholder column headers
        logger.info(f"Created training dataset file: {Config.TRAIN_DATA_PATH}")

    if not os.path.exists(Config.EVAL_DATA_PATH):
        with open(Config.EVAL_DATA_PATH, 'w') as f:
            f.write("text,answer\n")  # Placeholder column headers
        logger.info(f"Created evaluation dataset file: {Config.EVAL_DATA_PATH}")


class Chatbot:
    def __init__(self, model_path=None):
        """
        Initializes the chatbot with the specified model path or the default model.
        """
        self.model_manager = ModelManager()
        if model_path and os.path.exists(model_path):
            self.model_manager.load_model_from_path(model_path)
            logger.info(f"Loaded model from path: {model_path}")
        else:
            logger.warning(f"Model path '{model_path}' does not exist. Using default model '{Config.MODEL_NAME}'.")

    def ask(self, text, sampling=True, max_length=100, temperature=0.7, top_p=0.9, repetition_penalty=1.2):
        """
        Generates a response for a given input using the ModelManager.

        Args:
            text (str): Input text.
            sampling (bool): Whether to use sampling for text generation.
            max_length (int): Maximum length of the response.
            temperature (float): Sampling temperature.
            top_p (float): Nucleus sampling probability.
            repetition_penalty (float): Penalize repetitive tokens.

        Returns:
            str: The generated response.
        """
        try:
            return self.model_manager.generate_response(
                input_text=text,
                sampling=sampling,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm sorry, I couldn't process that."


class Application:
    def __init__(self, model_path=None):
        """
        Initializes the application with an optional model path.
        """
        self.chatbot = Chatbot(model_path=model_path)
        self.trainer_manager = TrainerManager(
            train_dataset_path=Config.TRAIN_DATA_PATH,
            eval_dataset_path=Config.EVAL_DATA_PATH
        )

    def train_model(self):
        """
        Fine-tunes the model using the training and evaluation datasets.
        """
        try:
            # Load datasets using DatasetLoader
            train_dataset = DatasetLoader.load_csv_to_dataset(Config.TRAIN_DATA_PATH)
            eval_dataset = DatasetLoader.load_csv_to_dataset(Config.EVAL_DATA_PATH)

            # Ensure datasets have enough entries
            if len(train_dataset) == 0 or len(eval_dataset) == 0:
                logger.error("Training or evaluation dataset is empty. Please check the dataset files.")
                return

            # Select first 100 questions for focused training
            train_dataset = train_dataset.select(range(min(100, len(train_dataset))))

            # Tokenize datasets
            train_dataset = DatasetLoader.preprocess_text(train_dataset, self.chatbot.model_manager.tokenizer)
            eval_dataset = DatasetLoader.preprocess_text(eval_dataset, self.chatbot.model_manager.tokenizer)

            # Assign processed datasets to trainer_manager
            self.trainer_manager.train_dataset = train_dataset
            self.trainer_manager.eval_dataset = eval_dataset

            # Start training
            self.trainer_manager.train()

            # Save trained model
            self.trainer_manager.save_trained_model(Config.MODEL_PATH)
            logger.info("Model training completed and saved.")

        except Exception as e:
            logger.error(f"Error during training: {e}")


    def interact(self):
        """
        Engages in a conversation loop with the user.
        """
        print("Welcome to the Chatbot! Type 'exit' or 'quit' to end the chat.")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting chat. Goodbye!")
                break

            # Generate a response
            response = self.chatbot.ask(
                text=user_input,
                sampling=True,
                max_length=200,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
            )

            # Save the interaction to training data
            self.trainer_manager.save_interaction(user_input, response, Config.TRAIN_DATA_PATH)

            print(f"AI: {response}")


def main():
    """
    Main entry point for the chatbot application.
    """
    # Ensure necessary directories and files exist
    ensure_directories_and_files()

    # Train the model before starting interaction
    app = Application()
    app.train_model()  # Train the model

    # Load the fine-tuned model and start interaction
    app = Application(model_path=Config.MODEL_PATH)
    app.interact()



if __name__ == "__main__":
    main()
