# LLM from Scratch

This project implements a complete autoregressive language model from first principles, including data preprocessing, tokenization, transformer architecture, training, and inference.

---

## Architecture

The model follows a decoder-only Transformer architecture.

---

## Embedding Layer

Given an input sequence:

<img src="https://latex.codecogs.com/svg.image?x=[x_1,x_2,\dots,x_T]" />

Token and positional embeddings:

<img src="https://latex.codecogs.com/svg.image?E_{tok}\in\mathbb{R}^{V\times&space;d},\quad&space;E_{pos}\in\mathbb{R}^{L\times&space;d}" />

Combined representation:

<img src="https://latex.codecogs.com/svg.image?\mathbf{h}_0=E_{tok}[x]+E_{pos}[:T]" />

---

## Transformer Block

Each block performs:

<img src="https://latex.codecogs.com/svg.image?\hat{\mathbf{h}}=\text{LayerNorm}(\mathbf{h})" />

<img src="https://latex.codecogs.com/svg.image?\mathbf{h}=\mathbf{h}+\text{MHA}(\hat{\mathbf{h}})" />

<img src="https://latex.codecogs.com/svg.image?\hat{\mathbf{h}}=\text{LayerNorm}(\mathbf{h})" />

<img src="https://latex.codecogs.com/svg.image?\mathbf{h}=\mathbf{h}+\text{FFN}(\hat{\mathbf{h}})" />

---

## Multi-Head Self-Attention

Query, Key, Value:

<img src="https://latex.codecogs.com/svg.image?\mathbf{Q}=\mathbf{h}W^Q,\quad\mathbf{K}=\mathbf{h}W^K,\quad\mathbf{V}=\mathbf{h}W^V" />

Scaled attention:

<img src="https://latex.codecogs.com/svg.image?A=\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_h}}" />

Causal masking:

<img src="https://latex.codecogs.com/svg.image?A_{ij}=-\infty\quad\text{if }i<j" />

Softmax:

<img src="https://latex.codecogs.com/svg.image?A=\text{softmax}(A)" />

Output:

<img src="https://latex.codecogs.com/svg.image?\mathbf{a}_h=A\mathbf{V}" />

Concatenation:

<img src="https://latex.codecogs.com/svg.image?\mathbf{a}=\text{concat}(\mathbf{a}_1,\dots,\mathbf{a}_H)W^O" />

---

## Feed-Forward Network

<img src="https://latex.codecogs.com/svg.image?\text{FFN}(\mathbf{x})=\max(0,\mathbf{x}W_1+b_1)W_2+b_2" />

---

## Output Layer

<img src="https://latex.codecogs.com/svg.image?\mathbf{h}_N=\text{LayerNorm}(\mathbf{h})" />

<img src="https://latex.codecogs.com/svg.image?\mathbf{l}=\mathbf{h}_NW_{head}+b_{head}" />

---

## Training

### Objective

<img src="https://latex.codecogs.com/svg.image?\mathcal{L}=-\sum_{t=1}^{T}\log&space;p(x_t|x_{<t})" />

Where:

<img src="https://latex.codecogs.com/svg.image?p(x_t|x_{<t})=\text{softmax}(\mathbf{l}_t)" />

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


## Text Generation

Autoregressive decoding:

<img src="https://latex.codecogs.com/svg.image?x_{t+1}\sim&space;P(x|x_{\leq&space;t})" />

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

## License

MIT

## Acknowledgments

Inspired by:
- Attention Is All You Need (Vaswani et al.)
- GPT-2 Paper (Radford et al.)
- Andrej Karpathy's nanoGPT
```

