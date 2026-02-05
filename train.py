import numpy as np
from pathlib import Path
from model import *

def main():
    text = Path("data/input.txt").read_text(encoding="utf-8")
    vocab = build_vocab(text)
    data = encode(text, vocab)
    
    learning_rate = 0.001
    iterations = 30000
    
    embed = InputEmbedding(vocab_size=vocab.size,context_length=32,dim=64)
    lnorm = LayerNorm(dim=64)
    attn = SelfAttention(dim=64)
    lnorm2 = LayerNorm(dim=64)
    ffn = FFN(dim=64, hidden_dim=4*64)
    proj = OutputProjection(dim=64,vocab_size=vocab.size)

    for i in range(iterations):
        x, y = make_batch(data, context_length=32, batch_size=12)
        x_emb = embed(x)

        # forward
        x_norm = lnorm.forward(x_emb)
        attn_out, attn_weights = attn.forward(x_norm)    
        x_res = x_norm + attn_out
        x_ffn_norm = lnorm2.forward(x_res)
        ffn_out = ffn.forward(x_ffn_norm)
        x_out = x_res + ffn_out
        logits = proj.forward(x_out)
        loss, probs = cross_entropy_loss(logits, y)
        
        ## Backwards
        dlogits = cross_entropy_backward(probs, y)
        dx_ffn = proj.backward(dlogits)  
        dx = ffn.backward(dx_ffn)
        dx_norm2 = lnorm2.backward(dx)
        dx_res = dx_norm2 + dx_ffn    
        dx_attn, _ = attn.backward(dx_res)
        dx_norm1 = lnorm.backward(dx_attn)
        dx_final = dx_norm1 + dx_res
        embed.backward(dx_final)
        
        #update weights
        attn.W_q -= learning_rate * attn.dW_q
        attn.W_k -= learning_rate * attn.dW_k
        attn.W_v -= learning_rate * attn.dW_v
        ffn.W1 -= learning_rate * ffn.dW1
        ffn.W2 -= learning_rate * ffn.dW2
        ffn.b1 -= learning_rate * ffn.db1
        ffn.b2 -= learning_rate * ffn.db2        
        lnorm.gamma -= learning_rate * lnorm.dgamma.squeeze()
        lnorm.beta -= learning_rate * lnorm.dbeta.squeeze()
        lnorm2.gamma -= learning_rate * lnorm2.dgamma.squeeze()
        lnorm2.beta -= learning_rate * lnorm2.dbeta.squeeze()
        proj.b -= learning_rate * proj.db
        proj.W -= learning_rate * proj.dW
        embed.token_emb.dW -= learning_rate * embed.token_emb.dW
        embed.pos_emb.dW -= learning_rate * embed.pos_emb.dW
     
        if i == 0 or i == iterations-1:
            print(f"Iteration {i}, Loss: {loss:.4f}")
        if i % 100 == 0:
            print("*", end="", flush=True)    



if __name__ == "__main__":
    main()
