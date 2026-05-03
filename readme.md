# LLM from Scratch

This project implements a complete language model from scratch, including data preprocessing, tokenization, model architecture, training, and inference.

## Architecture

The model follows the Transformer decoder architecture, similar to GPT models.

### Embedding Layer

The input sequence of tokens \( x = [x_1, x_2, \dots, x_T] \) is first embedded into a continuous vector space.

- **Token Embeddings**: \( E_{tok} \in \mathbb{R}^{V \times d} \), where \( V \) is vocabulary size and \( d \) is embedding dimension.
- **Positional Embeddings**: \( E_{pos} \in \mathbb{R}^{L \times d} \), where \( L \) is maximum sequence length.

The embedded representation is:
\[
\mathbf{h}_0 = E_{tok}[x] + E_{pos}[:T]
\]

### Transformer Block

The model consists of \( N \) identical transformer blocks. Each block performs:

1. **Layer Normalization**: \( \hat{\mathbf{h}} = \text{LayerNorm}(\mathbf{h}) \)
2. **Multi-Head Self-Attention**: \( \mathbf{a} = \text{MHA}(\hat{\mathbf{h}}) \)
3. **Residual Connection**: \( \mathbf{h} = \mathbf{h} + \mathbf{a} \)
4. **Layer Normalization**: \( \hat{\mathbf{h}} = \text{LayerNorm}(\mathbf{h}) \)
5. **Feed-Forward Network**: \( \mathbf{f} = \text{FFN}(\hat{\mathbf{h}}) \)
6. **Residual Connection**: \( \mathbf{h} = \mathbf{h} + \mathbf{f} \)

### Multi-Head Self-Attention

For each attention head \( h \):

- Query: \( \mathbf{Q} = \mathbf{h} W^Q \)
- Key: \( \mathbf{K} = \mathbf{h} W^K \)
- Value: \( \mathbf{V} = \mathbf{h} W^V \)

Attention weights:
\[
A = \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_h}}
\]

Causal masking (lower triangular):
\[
A_{ij} = -\infty \quad \text{if } i < j
\]

Softmax normalization:
\[
A = \softmax(A)
\]

Output:
\[
\mathbf{a}_h = A \mathbf{V}
\]

Multi-head concatenation and projection:
\[
\mathbf{a} = \concat(\mathbf{a}_1, \dots, \mathbf{a}_H) W^O
\]

### Feed-Forward Network

\[
\text{FFN}(\mathbf{x}) = \max(0, \mathbf{x} W_1 + b_1) W_2 + b_2
\]

Where \( W_1 \in \mathbb{R}^{d \times 4d} \), \( W_2 \in \mathbb{R}^{4d \times d} \).

### Final Layers

After \( N \) blocks:
\[
\mathbf{h}_N = \text{LayerNorm}(\mathbf{h}_{N-1})
\]

Logits:
\[
\mathbf{l} = \mathbf{h}_N W_{head} + b_{head}
\]

## Training

### Objective

Next-token prediction using cross-entropy loss:

\[
\mathcal{L} = -\sum_{t=1}^T \log p(x_t | x_{<t})
\]

Where \( p(x_t | x_{<t}) = \softmax(\mathbf{l}_t) \).

### Optimization

- Optimizer: AdamW with learning rate \( 10^{-4} \)
- Training steps: 5000
- Batch size: 32
- Sequence length: 128

## Tokenization

Uses Byte-Level Byte Pair Encoding (BPE):

- Vocabulary size: 256
- Special tokens: `<s>`, `<pad>`, `</s>`, `<unk>`

## Data Pipeline

1. **Data Preparation**:
   - Load CSV data
   - Clean text (remove extra whitespace)
   - Convert to text files

2. **Tokenization**:
   - Train BPE tokenizer on training data
   - Encode text to token IDs

3. **Training**:
   - Stream data in batches
   - Forward pass through model
   - Compute loss and backpropagate
   - Save model weights

4. **Inference**:
   - Load trained model
   - Encode input text
   - Generate tokens autoregressively
   - Decode to text

## Model Configuration

- Vocabulary size: 256
- Maximum sequence length: 128
- Embedding dimension: 512
- Number of attention heads: 8
- Number of layers: 10
- Feed-forward expansion: 4x

## Generation

Uses temperature sampling with optional top-k filtering:

- Temperature \( \tau \): scales logits before softmax
- Top-k: keeps only top k most probable tokens

Stops generation at end-of-sequence token.

## Usage

### Training

```bash
python -m src.training
```

### Inference

```bash
python -m app
```

## Dependencies

- torch
- tokenizers
- datasets
- tqdm