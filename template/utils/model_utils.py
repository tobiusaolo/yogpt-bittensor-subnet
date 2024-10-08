# model_utils.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from typing import List, Dict

class ModelManager:
    def __init__(self, model_name: str):
        """
        Initialize the ModelManager with specified model.
        
        Args:
            model_name (str): Name of the pre-trained model to load
        """
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Ensure the tokenizer has a padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
    
    def get_model_hash(self) -> str:
        """
        Generate a unique hash of model parameters.
        
        Returns:
            str: A hash string representing the current model state
        """
        return str(hash(str(self.model.state_dict())))
    
    def train_step(self, batch_texts: List[str], training_params: Dict) -> float:
        """
        Perform a single training step.
        
        Args:
            batch_texts (List[str]): List of text samples for training
            training_params (Dict): Dictionary containing training parameters
            
        Returns:
            float: The loss value for this training step
        """
        # Tokenize the batch
        inputs = self.tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Set up labels for causal language modeling
        inputs['labels'] = inputs['input_ids'].clone()
        
        # Forward pass
        self.model.train()
        outputs = self.model(**inputs)
        loss = outputs.loss
        
        return loss.item()
    
    def to(self, device: torch.device):
        """
        Move the model to the specified device.
        
        Args:
            device (torch.device): Device to move the model to
        """
        self.model.to(device)
    
    def save_state(self, path: str = "model_checkpoint.pt"):
        """
        Save the model's current state.
        
        Args:
            path (str): File path to save the model checkpoint.
        """
        torch.save(self.model.state_dict(), path)
    
    def load_state(self, path: str = "model_checkpoint.pt"):
        """
        Load the model state from a checkpoint.
        
        Args:
            path (str): File path to load the model checkpoint from.
        """
        self.model.load_state_dict(torch.load(path))
