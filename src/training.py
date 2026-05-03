from attention import LLM
import torch
import torch.nn.functional as F
from utils import encode_text

class dataStreamer:
    def __init__(self, file_path, block_size, batch_size, device):
        self.file_path = file_path
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.token_buffer = []
        self.position = 0
        
        # Pre-load first chunk
        self.load_more_data()
    
    def load_more_data(self):
        chunk_size = self.batch_size * self.block_size * 10  # Load 10 batches worth
        with open(self.file_path, "r", encoding="utf-8") as f:
            f.seek(self.position)
            chunk = f.read(chunk_size)
            if chunk:
                new_tokens = encode_text(chunk)
                self.token_buffer.extend(new_tokens)
                self.position += len(chunk)
    
    def get_batch(self):
        # Ensure we have enough data
        needed_tokens = self.batch_size * (self.block_size + 1)
        while len(self.token_buffer) < needed_tokens:
            self.load_more_data()
            if len(self.token_buffer) < needed_tokens:
                # Reset to beginning if not enough data
                self.position = 0
                self.token_buffer = []
                self.load_more_data()
        
        max_start = len(self.token_buffer) - self.block_size - 1
        ix = torch.randint(0, max_start, (self.batch_size,))
        
        x = torch.stack([
            torch.tensor(self.token_buffer[i:i+self.block_size], dtype=torch.long)
            for i in ix
        ])
        y = torch.stack([
            torch.tensor(self.token_buffer[i+1:i+self.block_size+1], dtype=torch.long)
            for i in ix
        ])
        
        return x.to(self.device), y.to(self.device)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loss_fn(logits,targets):
    B,T,C=logits.shape
    logits=logits.view(B*T,C)  #reshape to (B*T,vocab_size)
    targets=targets.view(B*T)   #reshape to (B*T)
    loss=F.cross_entropy(logits,targets)  #compute cross-entropy loss
    return loss



def train(model, device, steps=5000):

    data_streamer = dataStreamer("data/train.txt", block_size=128, batch_size=32, device=device)
    
    model.train()
    for step in range(steps):
        # get a batch of data
        x, y = data_streamer.get_batch()
        # forward pass
        logits = model(x)

        # compute loss
        loss = loss_fn(logits, y)

        # backward pass
        optimizer.zero_grad()
        loss.backward()

        # update weights
        optimizer.step()

        # logging
        if step % 100 == 0:
            print(f"step {step} | loss {loss.item():.4f}")

    # save model
    torch.save(model.state_dict(), "model/model.pt")

model=LLM(vocab_size=256,block_size=128,n_embed=512,num_heads=8,num_layers=10).to(device)  #initialize the model and move to device
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)  #initialize the optimizer

train(model,device)
