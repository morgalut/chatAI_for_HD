import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import Config
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, model_name=Config.MODEL_NAME, task="causal"):
        """
        Initializes the ModelManager for conversational tasks.
        Args:
            model_name (str): Name of the model to load.
            task (str): Task type, default is 'causal' for conversational tasks.
        """
        self.model_name = model_name
        self.task = task
        self.tokenizer = None
        self.model = None
        self.ensure_data_files_exist()
        self.load_model_and_tokenizer()

    def ensure_data_files_exist(self):
        """
        Ensures that necessary data files exist.
        If files or directories are missing, they will be created.
        """
        # Ensure model directory exists
        if not os.path.exists(Config.MODEL_PATH):
            os.makedirs(Config.MODEL_PATH, exist_ok=True)
            logger.info(f"Created model directory: {Config.MODEL_PATH}")

        # Ensure training data file exists
        if not os.path.exists(Config.TRAIN_DATA_PATH):
            with open(Config.TRAIN_DATA_PATH, 'w') as f:
                f.write("text,answer\n")  # Placeholder column headers
            logger.info(f"Created training data file: {Config.TRAIN_DATA_PATH}")

        # Ensure evaluation data file exists
        if not os.path.exists(Config.EVAL_DATA_PATH):
            with open(Config.EVAL_DATA_PATH, 'w') as f:
                f.write("text,answer\n")  # Placeholder column headers
            logger.info(f"Created evaluation data file: {Config.EVAL_DATA_PATH}")

    def load_model_and_tokenizer(self):
        """
        Loads the tokenizer and model based on the specified task or initializes them if missing.
        """
        if os.path.exists(Config.MODEL_PATH):
            try:
                logger.info(f"Attempting to load model from: {Config.MODEL_PATH}")
                self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_PATH)
                self.model = AutoModelForCausalLM.from_pretrained(Config.MODEL_PATH)
                logger.info("Model and tokenizer loaded successfully from saved path.")
                return
            except Exception as e:
                logger.warning(f"Failed to load model from {Config.MODEL_PATH}: {e}")

        logger.info(f"Loading default model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

        # Ensure the tokenizer has a padding token
        if self.tokenizer.pad_token is None:
            logger.info("Tokenizer does not have a padding token. Setting eos_token as pad_token.")
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Save the default model to the specified path for future use
        if not os.path.exists(Config.MODEL_PATH):
            os.makedirs(Config.MODEL_PATH, exist_ok=True)
        self.save_model(Config.MODEL_PATH)
        logger.info(f"Default model saved to: {Config.MODEL_PATH}")

    def prepare_input(self, text):
        """
        Prepares the input for the model.

        Args:
            text (str): Input text for tokenization.

        Returns:
            dict: Tokenized input for the model.
        """
        return self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=Config.MAX_INPUT_LENGTH,
        )

    def generate_response(
        self,
        input_text,
        sampling=True,
        max_length=100,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.0,
    ):
        """
        Generates a response for a given input.

        Args:
            input_text (str): The input text to generate a response for.
            sampling (bool): Whether to use sampling for text generation.
            max_length (int): Maximum length of the response.
            temperature (float): Sampling temperature (only if sampling=True).
            top_p (float): Nucleus sampling probability (only if sampling=True).
            repetition_penalty (float): Penalize repetitive tokens. Defaults to 1.0.

        Returns:
            str: The generated response.
        """
        inputs = self.prepare_input(input_text)

        # Default arguments for generation
        generation_args = {
            "max_length": max_length,
            "num_return_sequences": 1,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": sampling,
            "repetition_penalty": repetition_penalty,  # Add repetition penalty
        }

        if sampling:
            # Add sampling-specific parameters
            generation_args.update({
                "temperature": temperature,
                "top_p": top_p,
            })

        output_ids = self.model.generate(**inputs, **generation_args)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def save_model(self, path):
        """
        Saves the model and tokenizer to a specified path.
        """
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        logger.info(f"Saving model and tokenizer to {path}")
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_model_from_path(self, path):
        """
        Loads the model and tokenizer from a specified path.
        """
        if not os.path.exists(path):
            logger.warning(f"Model path '{path}' does not exist. Loading default model instead.")
            self.load_model_and_tokenizer()
            return

        try:
            logger.info(f"Loading model and tokenizer from {path}")
            self.tokenizer = AutoTokenizer.from_pretrained(path)

            # Ensure the tokenizer has a padding token
            if self.tokenizer.pad_token is None:
                logger.info("Tokenizer does not have a padding token. Setting eos_token as pad_token.")
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(path)
            logger.info("Model and tokenizer loaded successfully from the path.")
        except Exception as e:
            logger.error(f"Error loading model from path '{path}': {e}")
            raise
