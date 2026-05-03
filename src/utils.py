import pandas as pd
from tokenizers import ByteLevelBPETokenizer
import torch

tokenizer = ByteLevelBPETokenizer()

def clean_text(text):
    text=text.strip()
    text=text.replace("\n"," ")
    text=" ".join(text.split())
    return text




def convert_to_text():
    df_val=pd.read_csv("data/validation.csv")
    with open("data/validation.txt","w",encoding="utf-8") as f:  
        for i in range(len(df_val)):
            if(len(str(df_val["text"][i])))<20:
                continue
            f.write(clean_text(str(df_val["text"][i]))+"\n")

    df_train=pd.read_csv("data/train.csv")
    with open("data/train.txt","w",encoding="utf-8") as f:  
        for i in range(len(df_train)):
            if (len(str(df_train["text"][i])))<20:
                continue
            f.write(str(clean_text(df_train["text"][i]))+"\n")



def train_tokenizer():
    tokenizer.train(["data/train.txt"],vocab_size=256, min_frequency=2, special_tokens=["<s>","<pad>","</s>","<unk>"])
    tokenizer.save_model("tokenizer/")


def encode_text(text):
    tokenizer=ByteLevelBPETokenizer("tokenizer/vocab.json","tokenizer/merges.txt")
    return tokenizer.encode(text).ids

def decode_tokens(tokens):
    tokenizer=ByteLevelBPETokenizer("tokenizer/vocab.json","tokenizer/merges.txt")
    return tokenizer.decode(tokens)

