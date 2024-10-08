import time
import torch
import bittensor as bt
import os
import asyncio
import nest_asyncio
from typing import Tuple, Dict, Any
from torch.optim import AdamW
from tqdm import tqdm
from template.utils.data_utils import DataManager
from template.utils.model_utils import ModelManager
from template.base.miner import BaseMinerNeuron
from template.protocol import TrainingProtocol

# Apply nest_asyncio to allow asyncio event loops in threads
nest_asyncio.apply()

class TrainingMiner(BaseMinerNeuron):
    def __init__(
        self,
        config=None,
        model_type: str = 'gpt2',
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 1e-4
    ):
        super().__init__(config=config)
        
        # Training parameters
        self.model_type = model_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Initialize model manager and move model to the appropriate device
        self.model_manager = ModelManager(model_type)
        self.model_manager.to("cpu")
        
        # Initialize optimizer
        self.optimizer = AdamW(
            self.model_manager.model.parameters(),
            lr=self.learning_rate
        )
        
        # Initialize data manager
        self.data_manager = DataManager()
        
        # Training metrics storage
        self.training_history: Dict[str, Dict[str, Any]] = {}
        
        bt.logging.info(f"Initialized miner with model: {model_type}")
        bt.logging.info(f"Training parameters: epochs={epochs}, batch_size={batch_size}, learning_rate={learning_rate}")

    async def forward(self, synapse: TrainingProtocol) -> TrainingProtocol:
        """
        Processes the incoming synapse by performing training steps.
        
        Args:
            synapse (TrainingProtocol): The incoming synapse containing training parameters
            
        Returns:
            TrainingProtocol: The processed synapse containing training results
        """
        try:
            # Initialize training history for this run
            run_id = f"{self.model_type}_{time.strftime('%Y%m%d_%H%M%S')}"
            self.training_history[run_id] = {
                'epoch_losses': [],
                'avg_loss': 0.0,
                'best_loss': float('inf'),
                'start_time': time.time()
            }
            
            bt.logging.info(f"Starting training for {self.epochs} epochs...")
            
            # Calculate steps per epoch based on dataset size
            dataset_size = self.data_manager.get_dataset_size()
            steps_per_epoch = min(100, dataset_size // self.batch_size)
            
            for epoch in range(self.epochs):
                epoch_losses = []
                progress_bar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{self.epochs}")
                
                for _ in progress_bar:
                    # Get batch and perform training step
                    batch_texts = self.data_manager.get_batch(self.batch_size)
                    training_params = {'learning_rate': self.learning_rate}
                    
                    self.optimizer.zero_grad()
                    loss = self.model_manager.train_step(batch_texts, training_params)
                    self.optimizer.step()
                    
                    epoch_losses.append(loss)
                    
                    # Update progress bar
                    avg_loss = sum(epoch_losses) / len(epoch_losses)
                    progress_bar.set_postfix({
                        'loss': f"{loss:.4f}",
                        'avg_loss': f"{avg_loss:.4f}"
                    })
                
                # Epoch summary
                epoch_avg_loss = sum(epoch_losses) / len(epoch_losses)
                self.training_history[run_id]['epoch_losses'].append(epoch_avg_loss)
                self.training_history[run_id]['best_loss'] = min(
                    self.training_history[run_id]['best_loss'],
                    epoch_avg_loss
                )
                
                bt.logging.info(f"Epoch {epoch+1} completed - Avg Loss: {epoch_avg_loss:.4f}")
            
            # Calculate final metrics
            training_time = time.time() - self.training_history[run_id]['start_time']
            final_avg_loss = sum(self.training_history[run_id]['epoch_losses']) / len(self.training_history[run_id]['epoch_losses'])
            
            # Fill response
            synapse.loss = final_avg_loss
            synapse.model_hash = self.model_manager.get_model_hash()
            synapse.training_metrics = {
                'total_epochs': self.epochs,
                'best_loss': self.training_history[run_id]['best_loss'],
                'average_loss': final_avg_loss,
                'training_time_seconds': training_time
            }
            
            bt.logging.info(f"Training completed in {training_time:.2f} seconds")
            bt.logging.info(f"Final average loss: {final_avg_loss:.4f}")
            bt.logging.info(f"Best loss achieved: {self.training_history[run_id]['best_loss']:.4f}")
            
        except Exception as e:
            bt.logging.error(f"Error in forward: {str(e)}")
        
        return synapse

    async def blacklist(self, synapse: TrainingProtocol) -> Tuple[bool, str]:
        """Check if the incoming request should be blacklisted"""
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            return True, "Unrecognized hotkey"
        return False, "Hotkey recognized!"

    async def priority(self, synapse: TrainingProtocol) -> float:
        """Assign priority to incoming request based on caller's stake"""
        caller_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        priority = float(self.metagraph.S[caller_uid])
        return priority

    def save_state(self):
        """
        Save model state if needed.
        """
        self.model_manager.save_state("model_checkpoint.pt")

    def load_state(self):
        """
        Load model state if checkpoint exists.
        """
        self.model_manager.load_state("model_checkpoint.pt")


# This is the main function, which runs the miner.
if __name__ == "__main__":
    miner = TrainingMiner()
    try:
        # Ensure the miner is continuously running
        while True:
            bt.logging.info(f"Miner running... {time.time()}")
            time.sleep(30)
    except KeyboardInterrupt:
        bt.logging.info("Miner stopped.")
