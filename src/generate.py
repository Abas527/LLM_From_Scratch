import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from .utils import encode_text, decode_tokens
from .attention import LLM


eos_token_id=encode_text("</s>")[0] 
@torch.no_grad()
def generate(model, idx, max_new_tokens, block_size, temperature=0.5, top_k=None):

    model.eval()

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]  # (B,T)

        logits = model(idx_cond)  # (B,T,C)
        logits = logits[:, -1, :] / temperature  # (B,C)

        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, -1, None]] = -float('inf')

        probs = F.softmax(logits, dim=-1)  # (B,C)
        idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
        if idx_next.item() == eos_token_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)  # (B,T+1)
    return idx
def pipeline(text):
    model = LLM(vocab_size=256, block_size=128, n_embed=512, num_heads=8, num_layers=10)
    model.load_state_dict(torch.load("model/model.pt", map_location='cpu'))
    model.eval()

    idx = torch.tensor(encode_text(text)).unsqueeze(0)
    generated = generate(model, idx, max_new_tokens=1000, block_size=128)
    return decode_tokens(generated[0].tolist())



if __name__ == "__main__":
    input_text = "Once there was a little girl named Sarah."
    output_text = pipeline(input_text)
    print(output_text)