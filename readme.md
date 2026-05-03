# LLM from Scratch

This project implements a complete autoregressive language model from first principles, including data preprocessing, tokenization, transformer architecture, training, and inference.

---

## Architecture

The model follows a decoder-only Transformer architecture.

---

## Embedding Layer

Given an input sequence: x = [x1, x2, ..., xT]

Token and positional embeddings:
- Token Embeddings: E_tok in R^(V x d), where V = vocabulary size, d = embedding dimension
- Positional Embeddings: E_pos in R^(L x d), where L = maximum sequence length

Combined representation:
h0 = E_tok[x] + E_pos[:T]

---

## Transformer Block

Each block performs operations in the following order:

1. h_hat = LayerNorm(h)
2. a = MHA(h_hat)
3. h = h + a
4. h_hat = LayerNorm(h)
5. f = FFN(h_hat)
6. h = h + f

---

## Multi-Head Self-Attention

For each attention head:

Query: Q = h W^Q
Key: K = h W^K
Value: V = h W^V

Scaled attention scores: A = (Q K^T) / sqrt(d_h)

Causal masking: A_ij = -infinity if i < j

Softmax normalization: A = softmax(A)

Output for one head: a_h = A V

Multi-head concatenation and projection: a = concat(a1, ..., a_H) W^O

---

## Feed-Forward Network

FFN(x) = max(0, x W1 + b1) W2 + b2

Where W1 in R^(d x 4d) and W2 in R^(4d x d)

---

## Output Layer

After N transformer blocks:
h_N = LayerNorm(h_{N-1})

Logits: l = h_N W_head + b_head

---

## Training

### Objective

Next-token prediction using cross-entropy loss:
L = - sum_{t=1 to T} log p(x_t | x_<t)

Where:
p(x_t | x_<t) = softmax(l_t)

---

## Optimization

- Optimizer: AdamW
- Learning rate: 3e-4
- Batch size: 32
- Sequence length: 128

---

## Tokenization

Byte-Level BPE:
- Vocabulary size: 256
- Special tokens: `<s>`, `</s>`, `<pad>`, `<unk>`

---

## Data Pipeline

1. Load and clean text
2. Train tokenizer
3. Encode into token IDs
4. Sample random batches
5. Train using next-token prediction

---

## Text Generation

Autoregressive decoding: x_t ~ softmax(l_t / tau) where tau is temperature

Supports:
- Temperature sampling
- Top-k filtering
- Repetition penalty
- EOS stopping

---

## Model Configuration

Default configuration:
```python
{
    "vocab_size": 256,
    "max_seq_len": 128,
    "embed_dim": 512,
    "num_heads": 8,
    "num_layers": 10,
    "ffn_expansion": 4
}
```

## Usage

### Installation

```bash
# Clone the repository
git clone https://github.com/Abas527/LLM_From_Scratch
cd llm-from-scratch

# Install dependencies
pip install torch tokenizers datasets tqdm
```

### Training

```bash
# Train a new model
python -m src.training

# Or with custom parameters
python -m src.training --config configs/train.yaml
```

### Inference / Text Generation

```bash
# Run the interactive generation app
python -m app

# Or use the CLI
python generate.py --prompt "Once upon a time" --temperature 0.8 --top_k 40
```

## Project Structure

```
llm-from-scratch/
├── src/
│   ├── model.py          # Transformer model definition
│   ├── attention.py      # Multi-head attention module
│   ├── tokenizer.py      # BPE tokenizer implementation
│   ├── training.py       # Training loop and pipeline
│   └── utils.py          # Helper functions
├── app.py                # Inference interface
├── generate.py           # CLI generation script
├── configs/              # Configuration files
├── data/                 # Dataset storage
├── checkpoints/          # Trained model weights
└── README.md
```

## Dependencies

- torch
- tokenizers
- datasets
- tqdm

## Notes

- Designed for learning and experimentation
- Performance depends on dataset quality and training time
- Small models (~10M parameters) are useful for prototyping

## Future Improvements

- Flash Attention support for faster training
- Multi-GPU distributed training
- KV-caching for inference speedup
- Model checkpoint resuming
- Evaluation metrics (perplexity, accuracy)
- Fine-tuning interface
- Web UI for generation

## Acknowledgments

Inspired by:
- Attention Is All You Need (Vaswani et al.)
- GPT-2 Paper (Radford et al.)
- Andrej Karpathy's nanoGPT
```