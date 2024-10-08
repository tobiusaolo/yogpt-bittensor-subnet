import time
import torch
import numpy as np
import bittensor as bt

from template.base.validator import BaseValidatorNeuron
from template.protocol import TrainingProtocol
from template.utils.data_utils import DataManager

class TrainingValidator(BaseValidatorNeuron):
    def __init__(self, config=None):
        super().__init__(config=config)
        
        # Initialize data manager
        self.data_manager = DataManager("wikitext")  # Can be configured
        
    async def forward(self):
        """Query miners for training and evaluate their performance"""
        
        try:
            # Prepare batch
            batch_data = self.data_manager.get_batch(batch_size=32)
            
            # Query miners
            responses = await self.dendrite.query(
                self.metagraph.axons,
                TrainingProtocol(
                    model_name="gpt2",  # or 'llama2'
                    batch_data=batch_data,
                    training_params={"learning_rate": 1e-4}
                ),
                timeout=30.0
            )
            
            # Process responses
            rewards = torch.zeros(len(self.metagraph.axons))
            for i, resp in enumerate(responses):
                if resp is not None and resp.loss is not None:
                    rewards[i] = 1.0 / (1.0 + resp.loss)  # Higher reward for lower loss
            
            # Update scores
            uids = self.metagraph.uids.tolist()
            self.update_scores(rewards, uids)
            
        except Exception as e:
            bt.logging.error(f"Error in forward: {str(e)}")

# Main execution
if __name__ == "__main__":
    with TrainingValidator() as validator:
        while True:
            bt.logging.info(f"Validator running... {time.time()}")
            time.sleep(5)