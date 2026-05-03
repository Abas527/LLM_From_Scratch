import os
import requests

os.makedirs("pngs", exist_ok=True)

equations = [
    "x=[x_1,x_2,\\dots,x_T]",
    "E_{tok}\\in\\mathbb{R}^{V\\times d},\\quad E_{pos}\\in\\mathbb{R}^{L\\times d}",
    "\\mathbf{h}_0=E_{tok}[x]+E_{pos}[:T]",
    "\\hat{\\mathbf{h}}=\\text{LayerNorm}(\\mathbf{h})",
    "\\mathbf{h}=\\mathbf{h}+\\text{MHA}(\\hat{\\mathbf{h}})",
    "\\hat{\\mathbf{h}}=\\text{LayerNorm}(\\mathbf{h})",
    "\\mathbf{h}=\\mathbf{h}+\\text{FFN}(\\hat{\\mathbf{h}})",
    "\\mathbf{Q}=\\mathbf{h}W^Q,\\quad\\mathbf{K}=\\mathbf{h}W^K,\\quad\\mathbf{V}=\\mathbf{h}W^V",
    "A=\\frac{\\mathbf{Q}\\mathbf{K}^\\top}{\\sqrt{d_h}}",
    "A_{ij}=-\\infty\\quad\\text{if }i<j",
    "A=\\text{softmax}(A)",
    "\\mathbf{a}_h=A\\mathbf{V}",
    "\\mathbf{a}=\\text{concat}(\\mathbf{a}_1,\\dots,\\mathbf{a}_H)W^O",
    "\\text{FFN}(\\mathbf{x})=\\max(0,\\mathbf{x}W_1+b_1)W_2+b_2",
    "\\mathbf{h}_N=\\text{LayerNorm}(\\mathbf{h})",
    "\\mathbf{l}=\\mathbf{h}_NW_{head}+b_{head}",
    "\\mathcal{L}=-\\sum_{t=1}^{T}\\log p(x_t|x_{<t})",
    "p(x_t|x_{<t})=\\text{softmax}(\\mathbf{l}_t)",
    "x_{t+1}\\sim P(x|x_{\\leq t})"
]

base_url = "https://latex.codecogs.com/png.image?"

for i, eq in enumerate(equations):
    url = base_url + eq
    r = requests.get(url)

    with open(f"assets/svg{i}.png", "wb") as f:
        f.write(r.content)

    print(f"Downloaded PNG {i}")

print("All PNGs downloaded.")