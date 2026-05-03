---

# LLM from Scratch

This project implements a complete autoregressive language model from first principles, including data preprocessing, tokenization, transformer architecture, training pipeline, and inference. The implementation follows a decoder-only Transformer design inspired by modern generative language models.

---

## Architecture

The model is a causal (decoder-only) Transformer composed of token embeddings, positional encodings, stacked transformer blocks, and a final projection head.

### Embedding Layer

Given an input token sequence:

[
x = [x_1, x_2, \dots, x_T]
]

* Token embeddings: ( E_{tok} \in \mathbb{R}^{V \times d} )
* Positional embeddings: ( E_{pos} \in \mathbb{R}^{L \times d} )

The input representation is:

[
\mathbf{h}*0 = E*{tok}[x] + E_{pos}[:T]
]

---

### Transformer Block

Each block applies:

1. Layer Normalization
2. Multi-Head Self-Attention (causal)
3. Residual connection
4. Layer Normalization
5. Feed-Forward Network
6. Residual connection

Formally:

[
\hat{\mathbf{h}} = \text{LayerNorm}(\mathbf{h})
]
[
\mathbf{h} = \mathbf{h} + \text{MHA}(\hat{\mathbf{h}})
]
[
\hat{\mathbf{h}} = \text{LayerNorm}(\mathbf{h})
]
[
\mathbf{h} = \mathbf{h} + \text{FFN}(\hat{\mathbf{h}})
]

---

### Multi-Head Self-Attention

For each head:

[
\mathbf{Q} = \mathbf{h} W^Q, \quad
\mathbf{K} = \mathbf{h} W^K, \quad
\mathbf{V} = \mathbf{h} W^V
]

Scaled dot-product attention:

[
A = \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_h}}
]

Causal masking ensures autoregressive behavior:

[
A_{ij} = -\infty \quad \text{if } i < j
]

[
A = \text{softmax}(A)
]

[
\mathbf{a}_h = A \mathbf{V}
]

Outputs from all heads are concatenated and projected:

[
\mathbf{a} = \text{concat}(\mathbf{a}_1, \dots, \mathbf{a}_H) W^O
]

---

### Feed-Forward Network

[
\text{FFN}(\mathbf{x}) = \max(0, \mathbf{x} W_1 + b_1) W_2 + b_2
]

Where:

* ( W_1 \in \mathbb{R}^{d \times 4d} )
* ( W_2 \in \mathbb{R}^{4d \times d} )

---

### Output Layer

After the final transformer block:

[
\mathbf{h}_N = \text{LayerNorm}(\mathbf{h})
]

Logits over vocabulary:

[
\mathbf{l} = \mathbf{h}*N W*{head} + b_{head}
]

---

## Training

### Objective

The model is trained using next-token prediction with cross-entropy loss:

[
\mathcal{L} = -\sum_{t=1}^{T} \log p(x_t \mid x_{<t})
]

---

### Optimization

* Optimizer: AdamW
* Learning rate: ( 3 \times 10^{-4} )
* Batch size: 32
* Sequence length: 128
* Training steps: configurable

---

## Tokenization

Byte-Level Byte Pair Encoding (BPE) is used:

* Vocabulary size: 256
* Special tokens:

  * `<s>` (start)
  * `</s>` (end)
  * `<pad>`
  * `<unk>`

---

## Data Pipeline

1. Data Preparation

   * Load raw text or structured data
   * Normalize whitespace and formatting

2. Tokenization

   * Train BPE tokenizer
   * Encode text into token IDs

3. Batching

   * Randomly sample contiguous token blocks
   * Create input-target pairs with one-token shift

4. Training

   * Forward pass through model
   * Compute cross-entropy loss
   * Backpropagation and parameter updates
   * Periodic checkpointing

5. Inference

   * Encode prompt
   * Generate tokens autoregressively
   * Decode tokens back to text

---

## Model Configuration

* Vocabulary size: 256
* Context length: 128
* Embedding dimension: 512
* Number of attention heads: 8
* Number of transformer layers: 6–10
* Feed-forward expansion: 4×

---

## Text Generation

Supports autoregressive decoding with:

* Temperature scaling
* Top-k sampling
* Repetition penalty
* End-of-sequence stopping

Optional decoding strategies include beam search for improved coherence.

---

## Usage

### Training

```bash
python -m src.training
```

### Inference

```bash
python -m app
```

---

## Dependencies

* torch
* tokenizers
* datasets
* tqdm

---

## Notes

* The implementation is designed for educational and experimental purposes.
* The model size (~10M parameters) is suitable for small-scale tasks and prototyping.
* Performance depends heavily on dataset quality and training duration.

---

