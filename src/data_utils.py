import torch
from config import SEQUENCE_LENGTH, DEVICE


def load_poems(file_path='./data/turkish_poems.txt'):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text


def create_vocabulary(text):
    characters = sorted(list(set(text)))
    vocab_size = len(characters)
    char_to_index = {ch: i for i, ch in enumerate(characters)}
    index_to_char = {i: ch for i, ch in enumerate(characters)}
    return characters, vocab_size, char_to_index, index_to_char


def encode_text(text, char_to_index):
    return [char_to_index[c] for c in text]


def decode_text(encoded_text, index_to_char):
    return ''.join([index_to_char[i] for i in encoded_text])


def prepare_dataset(encoded_text):
    data = torch.tensor(encoded_text, dtype=torch.long)
    split_index = int(0.9 * len(data))
    train_data = data[:split_index]
    val_data = data[split_index:]
    return train_data, val_data


def get_batch(data, val_data, split, batch_size, sequence_length):
    data_split = data if split == 'train' else val_data
    indices = torch.randint(len(data_split) - sequence_length, (batch_size,))
    x = torch.stack([data_split[i:i + sequence_length] for i in indices])
    y = torch.stack([data_split[i + 1:i + sequence_length + 1]
                    for i in indices])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y
