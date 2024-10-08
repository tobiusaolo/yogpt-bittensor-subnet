import typing
import bittensor as bt
from typing import List,Optional
from transformers import PreTrainedTokenizer

class TrainingProtocol(bt.Synapse):
    """Protocol for GPT model training in Bittensor network"""
    
    # Required request fields
    model_name: str  # 'gpt2' or 'llama2'
    batch_data: List[str]  # Training data batch
    training_params: dict  # Hyperparameters
    
    # Response fields
    loss: Optional[float] = None
    model_hash: Optional[str] = None
    
    def deserialize(self) -> 'TrainingProtocol':
        return self
