import torch
import torch.nn as nn
import torch.nn.functional as F
from config import SEQUENCE_LENGTH, DROPOUT_RATE, EMBEDDING_DIM, NUM_ATTENTION_HEADS, NUM_TRANSFORMER_LAYERS, DEVICE


class AttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(EMBEDDING_DIM, head_size, bias=False)
        self.query = nn.Linear(EMBEDDING_DIM, head_size, bias=False)
        self.value = nn.Linear(EMBEDDING_DIM, head_size, bias=False)
        self.register_buffer('tril', torch.tril(
            torch.ones(SEQUENCE_LENGTH, SEQUENCE_LENGTH)))
        self.dropout = nn.Dropout(DROPOUT_RATE)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(head_size)
                                   for _ in range(num_heads)])
        self.projection = nn.Linear(head_size * num_heads, EMBEDDING_DIM)
        self.dropout = nn.Dropout(DROPOUT_RATE)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, 4 * EMBEDDING_DIM),
            nn.ReLU(),
            nn.Linear(4 * EMBEDDING_DIM, EMBEDDING_DIM),
            nn.Dropout(DROPOUT_RATE),
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = EMBEDDING_DIM // NUM_ATTENTION_HEADS
        self.attention = MultiHeadAttention(NUM_ATTENTION_HEADS, head_size)
        self.feed_forward = FeedForward()
        self.layer_norm1 = nn.LayerNorm(EMBEDDING_DIM)
        self.layer_norm2 = nn.LayerNorm(EMBEDDING_DIM)

    def forward(self, x):
        x = x + self.attention(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x


class TurkishPoemGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, EMBEDDING_DIM)
        self.position_embedding = nn.Embedding(SEQUENCE_LENGTH, EMBEDDING_DIM)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock() for _ in range(NUM_TRANSFORMER_LAYERS)])
        self.layer_norm_final = nn.LayerNorm(EMBEDDING_DIM)
        self.language_model_head = nn.Linear(EMBEDDING_DIM, vocab_size)
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, indices, targets=None):
        B, T = indices.shape
        token_embeddings = self.token_embedding(indices)
        position_embeddings = self.position_embedding(
            torch.arange(T, device=DEVICE))
        x = token_embeddings + position_embeddings
        x = self.transformer_blocks(x)
        x = self.layer_norm_final(x)
        logits = self.language_model_head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, indices, max_new_tokens):
        for _ in range(max_new_tokens):
            indices_cond = indices[:, -SEQUENCE_LENGTH:]
            logits, _ = self(indices_cond)
            logits = logits[:, -1, :]
            probabilities = F.softmax(logits, dim=-1)
            next_index = torch.multinomial(probabilities, num_samples=1)
            indices = torch.cat((indices, next_index), dim=1)
        return indices
