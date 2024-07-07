# TurkishPoemGPT Project

This project provides a simple implementation of a GPT-like model using PyTorch, specifically designed for generating Turkish poems. The project includes data loading, model training, and text generation functionalities. It is built on Carpathy's NanoGPT.

## Project Structure

```
turkishpoemgpt_project/
│
├── data/
│   ├── turkish_poems.txt
├── src/
│   ├── config.py
│   ├── data_utils.py
│   ├── model.py
│   ├── train.py
│   └── generate.py
├── requirements.txt
└── README.md
```

## Setup

1. Clone the repository:

2. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

3. Place your training data in the `data` directory. The data should be a plain text file named `turkish_poems.txt`.

## Usage

### Training

To train the model, run:

```
python src/train.py
```

### Text Generation

To generate text using the trained model, run:

```
python src/generate.py
```
