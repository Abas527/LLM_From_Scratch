import os
import requests
import zipfile

# Create folder
os.makedirs("assets", exist_ok=True)

# LaTeX equations (in order)
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

base_url = "https://latex.codecogs.com/svg.image?"

# Download SVGs
for i, eq in enumerate(equations):
    url = base_url + eq
    response = requests.get(url)
    
    file_path = f"assets/svg{i}.svg"
    with open(file_path, "wb") as f:
        f.write(response.content)

print("SVGs downloaded.")
