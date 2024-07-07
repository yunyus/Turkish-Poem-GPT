import torch
from data_utils import load_poems, create_vocabulary, encode_text, decode_text
from model import TurkishPoemGPT
from config import DEVICE


def generate_text(max_new_tokens=500):
    text = load_poems()
    characters, vocab_size, char_to_index, index_to_char = create_vocabulary(
        text)
    encoded_text = encode_text(text, char_to_index)

    print(f"Using device: {DEVICE}")
    model = TurkishPoemGPT(vocab_size).to(DEVICE)
    model.load_state_dict(torch.load("turkishpoemgpt_model.pth"))

    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    generated_text = decode_text(model.generate(
        context, max_new_tokens=max_new_tokens)[0].tolist(), index_to_char)
    print(generated_text)


if __name__ == "__main__":
    generate_text()
