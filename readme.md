# LLM from Scratch

This project implements a complete autoregressive language model from first principles, including data preprocessing, tokenization, transformer architecture, training, and inference.

---

## Architecture

The model follows a decoder-only Transformer architecture.

---

## Embedding Layer

Given an input sequence:

<img src="./assets/svg0.svg" />

Token and positional embeddings:

<img src="assets/svg1.svg" />

Combined representation:

<img src="assets/svg2.svg" />

---

## Transformer Block

Each block performs:

<img src="assets/svg3.svg" />

<img src="assets/svg4.svg" />

<img src="assets/svg5.svg" />

<img src="assets/svg6.svg" />

---

## Multi-Head Self-Attention

Query, Key, Value:

<img src="assets/svg7.svg" />

Scaled attention:

<img src="assets/svg8.svg" />

Causal masking:

<img src="assets/svg9.svg" />

Softmax:

<img src="assets/svg10.svg" />

Output:

<img src="assets/svg11.svg" />

Concatenation:

<img src="assets/svg12.svg" />

---

## Feed-Forward Network

<img src="assets/svg13.svg" />

---

## Output Layer

<img src="assets/svg14.svg" />

<img src="assets/svg15.svg" />

---

## Training

### Objective

<img src="assets/svg16.svg" />

Where:

<img src="assets/svg17.svg" />

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

Autoregressive decoding:

<img src="assets/svg18.svg" />

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