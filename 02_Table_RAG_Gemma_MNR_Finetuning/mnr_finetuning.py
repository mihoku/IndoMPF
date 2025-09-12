import json
import math
import logging
import sys
import torch
import pickle
import random
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from torch.utils.data import DataLoader
from tqdm import tqdm
from sentence_transformers.evaluation import SentenceEvaluator, SequentialEvaluator
import os

# --- 1. Configuration ---
MODEL_NAME = './gemma3.embedding'
TRAIN_FILE = 'data/mnr_training_data.pkl'
VALIDATION_FILE = 'data/mnr_validation_data.pkl'
OUTPUT_PATH = './final-mnr-finetuned-model' 
CHECKPOINT_PATH = './mnr-checkpoints'
SEED = 42 # The seed for reproducibility

TRAIN_BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 1e-5

logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)

# --- NEW: Function to set the seed for reproducibility ---
def set_seed(seed: int):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_pairs_from_pickle(file_path):
    """Loads a list of InputExample pairs from a pickle file."""
    try:
        with open(file_path, "rb") as f:
            pairs = pickle.load(f)
        logging.info(f"Loaded {len(pairs)} pairs from {file_path}")
        return pairs
    except FileNotFoundError:
        logging.error(f"Error: Data file '{file_path}' not found.")
        exit()

class LossEvaluator(SentenceEvaluator):
    # Evaluate the finetuned model
    def __init__(self, dataloader: DataLoader, loss_model, name: str = ''):
        self.dataloader = dataloader
        self.loss_model = loss_model
        self.name = name

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        self.loss_model.eval()
        total_loss = 0
        num_batches = 0
        device = model.device 

        with torch.no_grad():
            for features, labels in self.dataloader:
                for i in range(len(features)):
                    for key, value in features[i].items():
                        if isinstance(value, torch.Tensor):
                            features[i][key] = value.to(device)
                if isinstance(labels, torch.Tensor):
                    labels = labels.to(device)
                loss_value = self.loss_model(features, labels)
                total_loss += loss_value.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        logging.info(f"Evaluator '{self.name}' - Epoch {epoch}, Steps {steps} - Loss: {avg_loss:.4f}")
        return -avg_loss

# --- Main Training Script ---
if __name__ == "__main__":
    # --- NEW: Set the seed at the very beginning ---
    set_seed(SEED)

    # Load the pre-processed training and validation pairs
    train_examples = load_pairs_from_pickle(TRAIN_FILE)
    val_examples = load_pairs_from_pickle(VALIDATION_FILE)
    
    word_embedding_model = models.Transformer(MODEL_NAME, model_args={'device_map': 'auto'})
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    
    # The DataLoader will now use the seeded RNG for shuffling
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=TRAIN_BATCH_SIZE)
    
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # Setup evaluators
    logging.info("Preparing evaluators for training and validation sets...")
    train_eval_dataloader = DataLoader(train_examples, shuffle=False, batch_size=TRAIN_BATCH_SIZE, collate_fn=model.smart_batching_collate)
    train_evaluator = LossEvaluator(dataloader=train_eval_dataloader, loss_model=train_loss, name='train-eval')

    val_dataloader = DataLoader(val_examples, shuffle=False, batch_size=TRAIN_BATCH_SIZE, collate_fn=model.smart_batching_collate)
    val_evaluator = LossEvaluator(dataloader=val_dataloader, loss_model=train_loss, name='validation-eval')
    
    sequential_evaluator = SequentialEvaluator(
        evaluators=[train_evaluator, val_evaluator], 
        main_score_function=lambda scores: scores[-1]
    )

    warmup_steps = math.ceil(len(train_dataloader) * NUM_EPOCHS * 0.1)

    logging.info("Starting model fine-tuning with MultipleNegativesRankingLoss...")
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=sequential_evaluator,
        epochs=NUM_EPOCHS,
        evaluation_steps=len(train_dataloader),
        warmup_steps=warmup_steps,
        optimizer_params={'lr': LEARNING_RATE},
        output_path=OUTPUT_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        checkpoint_save_steps=len(train_dataloader),
        show_progress_bar=True,
    )