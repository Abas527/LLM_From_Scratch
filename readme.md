
```markdown
# LLM from Scratch

A complete implementation of a language model built from scratch, including data preprocessing, tokenization, transformer architecture, training pipeline, and inference.

## Features

- Transformer Decoder architecture inspired by GPT models
- Byte-Level BPE tokenization with configurable vocabulary
- Efficient training with streaming data pipeline
- Temperature and top-k sampling for controlled text generation
- Configurable model parameters (layers, heads, dimensions)

## Architecture

The model implements a standard Transformer decoder with causal attention masking.

### Embedding Layer

Input tokens `x = [x_1, x_2, ..., x_T]` are mapped to continuous vectors:

- **Token Embeddings**: `E_tok in R^(V×d)` where `V` = vocabulary size, `d` = embedding dimension
- **Positional Embeddings**: `E_pos in R^(L×d)` where `L` = maximum sequence length

The combined representation:
```
h_0 = E_tok[x] + E_pos[:T]
```

### Transformer Block

Each of the `N` identical blocks processes the input through:

```
h <- h + MHA(LayerNorm(h))
h <- h + FFN(LayerNorm(h))
```

### Multi-Head Self-Attention

For each head `h`:

1. Compute projections: `Q = hW^Q`, `K = hW^K`, `V = hW^V`
2. Calculate attention scores: `A = (QK^T) / sqrt(d_h)`
3. Apply causal masking: `A_ij = -inf if i < j`
4. Normalize with softmax: `A = softmax(A)`
5. Weight values: `a_h = A V`

Multiple heads are concatenated and projected:
```
a = concat(a_1, ..., a_H) W^O
```

### Feed-Forward Network

A two-layer MLP with ReLU activation:
```
FFN(x) = max(0, xW_1 + b_1) W_2 + b_2
```
- `W_1 in R^(d×4d)`, `W_2 in R^(4d×d)`

### Output Layer

Final processing:
```
h_N = LayerNorm(h_{N-1})
l = h_N W_head + b_head  # logits
```

## Training

### Objective

Next-token prediction using cross-entropy loss:
```
L = -sum_{t=1}^T log p(x_t | x_<t)
```
where `p(x_t | x_<t) = softmax(l_t)`

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Training steps | 5,000 |
| Batch size | 32 |
| Sequence length | 128 |

## Tokenization

Uses **Byte-Level BPE** (Byte Pair Encoding):
- Base vocabulary: 256 bytes
- Special tokens: `<s>`, `<pad>`, `</s>`, `<unk>`
- Trained on the target corpus

## Data Pipeline

```
CSV Data -> Text Cleaning -> Text Files -> BPE Training -> Tokenization -> Training -> Model -> Inference
```

1. **Data Preparation**: Load CSV, clean whitespace, save as text files
2. **Tokenization**: Train BPE tokenizer, encode text to IDs
3. **Training**: Stream batches -> forward pass -> loss computation -> backpropagation
4. **Inference**: Load model -> encode prompt -> generate tokens -> decode to text

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

## Text Generation

Supports sampling strategies for controlled generation:

- **Temperature scaling**: `tau` controls randomness (lower = more deterministic)
- **Top-k filtering**: Only keep `k` most probable tokens

Generation stops when end-of-sequence token (`</s>`) is produced.

## Usage

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-from-scratch
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

### Using as a Library

```python
from src.model import LLM
from src.tokenizer import BPETokenizer

# Load model
model = LLM.from_pretrained("checkpoints/model.pt")
tokenizer = BPETokenizer.load("tokenizer.json")

# Generate text
prompt = "The future of AI is"
input_ids = tokenizer.encode(prompt)
output_ids = model.generate(input_ids, max_length=100, temperature=0.7)
generated_text = tokenizer.decode(output_ids)

print(generated_text)
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

- `torch` - Deep learning framework
- `tokenizers` - Fast BPE tokenization
- `datasets` - Data loading utilities  
- `tqdm` - Progress bars

## Future Improvements

- Flash Attention support for faster training
- Multi-GPU distributed training
- KV-caching for inference speedup
- Model checkpoint resuming
- Evaluation metrics (perplexity, accuracy)
- Fine-tuning interface
- Web UI for generation

## License

MIT

## Acknowledgments

Inspired by:
- Attention Is All You Need (Vaswani et al.)
- GPT-2 Paper (Radford et al.)
- Andrej Karpathy's nanoGPT
```

This version maintains all the improvements from the previous version but removes all emoji characters, using clean text formatting instead.