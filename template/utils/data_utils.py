# data_utils.py
from typing import List
from datasets import load_dataset
import os

class DataManager:
    def __init__(self, dataset_name: str = "wikitext", subset: str = "wikitext-2-raw-v1"):
        """
        Initialize the DataManager with specified dataset, using Hugging Face authentication if required.
        
        Args:
            dataset_name (str): Name of the dataset to load
            subset (str): Specific subset of the dataset
        """
        # Get Hugging Face token from environment variable or prompt user to set it
        self.hf_token = "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"
        if not self.hf_token:
            raise ValueError("Hugging Face token is required! Set it via environment variable HUGGINGFACE_TOKEN.")
        

        self.dataset = load_dataset("carlosejimenez/wikitext__wikitext-2-raw-v1", token=self.hf_token)
        self.current_index = 0
        self.dataset_size = len(self.dataset['train'])
        
    def get_batch(self, batch_size: int) -> List[str]:
        """
        Get a batch of training data.
        
        Args:
            batch_size (int): Size of the batch to return
            
        Returns:
            List[str]: A list of text samples
        """
        texts = []
        for _ in range(batch_size):
            if self.current_index >= self.dataset_size:
                self.current_index = 0  # Reset if we've gone through the dataset
            
            text = self.dataset['train'][self.current_index]['text']
            if text.strip():  # Only add non-empty texts
                texts.append(text)
            
            self.current_index += 1
        
        return texts
    
    def get_dataset_size(self) -> int:
        """
        Get the total size of the training dataset.
        
        Returns:
            int: Number of samples in the training dataset
        """
        return self.dataset_size
