import torch

# Configuration and hyperparameters
BATCH_SIZE = 64
SEQUENCE_LENGTH = 256
MAX_ITERATIONS = 5000
EVALUATION_INTERVAL = 500
LEARNING_RATE = 3e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EVALUATION_ITERATIONS = 200
EMBEDDING_DIM = 384
NUM_ATTENTION_HEADS = 6
NUM_TRANSFORMER_LAYERS = 6
DROPOUT_RATE = 0.2

torch.manual_seed(1337)
