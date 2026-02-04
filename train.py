import numpy as np
from pathlib import Path
from model import build_vocab, encode, decode, make_batch, InputEmbedding, LayerNorm,SelfAttention,FeedForward

def main():
    text = Path("data/input.txt").read_text(encoding="utf-8")
    vocab = build_vocab(text)
    data = encode(text, vocab)

    x, y = make_batch(data, context_length=32, batch_size=2)
    
    embed = InputEmbedding(
        vocab_size=vocab.size,
        context_length=32,
        dim=64
    )

    x_emb = embed(x)

    lnorm = LayerNorm(dim=64)
    x_norm = lnorm.forward(x_emb)
    
    attn = SelfAttention(dim=64)
    attn_out, attn_weights = attn.forward(x_norm)
    
    x_res = x_norm + attn_out
    
    lnorm2 = LayerNorm(dim=64)
    ffn = FeedForward(dim=64)

    x_ffn_norm = lnorm2.forward(x_res)
    ffn_out = ffn.forward(x_ffn_norm)

    x_out = x_res + ffn_out


if __name__ == "__main__":
    main()
