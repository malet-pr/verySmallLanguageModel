import numpy as np
from pathlib import Path
import pickle
from model import *
import math


# 1. LOAD VOCAB
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)
stoi, itos = vocab['stoi'], vocab['itos']

# 2. LOAD WEIGHTS
weights = np.load("java_expert.npz")

# config dictionary
config = {
    "vocab_size": None,
    "dim": 512,              
    "context_length": 256,    
    "batch_size": 12,
    "lr": 3e-4,
    "n_heads": 8          
}

epochs = 200

# 3. INITIALIZE YOUR ARCHITECTURE (Use the OLD dimensions here!)
embed = InputEmbedding(vocab_size=len(stoi),context_length=config["context_length"],dim=config["dim"])
lnorm = LayerNorm(dim=config["dim"])
mhattn = MultiHeadAttention(d_model=config['dim'],num_heads=config['n_heads'])
lnorm2 = LayerNorm(dim=config["dim"])
ffn = FFN(dim=config["dim"], hidden_dim=4*config["dim"])
proj = OutputProjection(dim=config["dim"],vocab_size=len(stoi))

# 4. INJECT THE SAVED WEIGHTS
embed.token_emb.weight = weights['token_emb']
embed.pos_emb.weight = weights['pos_emb']
mhattn.w_q, mhattn.w_k, mhattn.w_v, mhattn.w_o = weights['w_q'], weights['w_k'], weights['w_v'],weights['w_o']
ffn_w1,ffn_b1,ffn_w2,ffn_b2 = weights['ffn_w1'],weights['ffn_b1'],weights['ffn_w2'],weights['ffn_b2']
lnorm.gamma,lnorm.beta,lnorm2.gamma,lnorm2.beta = weights['ln1_g'],weights['ln1_b'],weights['ln2_g'],weights['ln2_b']
proj.W,proj.b = weights['proj_w'],weights['proj_b']

text = Path("data/quarkus.txt").read_text(encoding="utf-8")
data = encode(text, vocab)

for i in range(epochs):
    x, y = make_batch(data, context_length=config["context_length"], batch_size=config["batch_size"])
    x_emb = embed(x)
    
    # Learning rate decay
    min_lr = config['lr'] * 0.1
    current_lr = min_lr + 0.5 * (config['lr'] - min_lr) * (1 + math.cos(math.pi * i / epochs))

    # forward
    x_norm = lnorm.forward(x_emb)
    mhattn_out = mhattn.forward(x_norm)   
    x_res = x_emb + mhattn_out
    x_ffn_norm = lnorm2.forward(x_res)
    ffn_out = ffn.forward(x_ffn_norm)
    x_out = x_res + ffn_out
    logits = proj.forward(x_out)
    loss, probs = cross_entropy_loss(logits, y)
    
    ## Backwards
    dlogits = cross_entropy_backward(probs, y)
    dout = proj.backward(dlogits)  
    dx_ffn = ffn.backward(dout)
    dx_norm2 = lnorm2.backward(dx_ffn)
    dx_res = dx_norm2 + dout    
    # dx_attn, _ = attn.backward(dx_res)
    # dx_norm1 = lnorm.backward(dx_attn)
    dx_mhattn = mhattn.backward(dx_res)
    dx_norm1 = lnorm.backward(dx_mhattn)
    dx_final = dx_norm1 + dx_res
    embed.backward(dx_final)
    
    #update weights
    mhattn.w_q -= current_lr * mhattn.d_wq
    mhattn.w_k -= current_lr * mhattn.d_wk
    mhattn.w_v -= current_lr * mhattn.d_wv
    mhattn.w_o -= current_lr * mhattn.d_wo
    ffn.W1 -= current_lr * ffn.dW1
    ffn.W2 -= current_lr * ffn.dW2
    ffn.b1 -= current_lr * ffn.db1
    ffn.b2 -= current_lr * ffn.db2        
    lnorm.gamma -= current_lr * lnorm.dgamma.squeeze()
    lnorm.beta -= current_lr * lnorm.dbeta.squeeze()
    lnorm2.gamma -= current_lr * lnorm2.dgamma.squeeze()
    lnorm2.beta -= current_lr * lnorm2.dbeta.squeeze()
    proj.b -= current_lr * proj.db
    proj.W -= current_lr * proj.dW
    embed.token_emb.weight -= current_lr * embed.token_emb.dW
    embed.pos_emb.weight -= current_lr * embed.pos_emb.dW
    
    if i%25 == 0 or i == epochs-1:
        print(f"Epoch {i}, Loss: {loss:.4f}")

# After the loop...
# Create a dictionary of all your trained weights
state_dict = {
    # Embeddings
    'token_emb': embed.token_emb.weight, 
    'pos_emb': embed.pos_emb.weight,
    # Attention
    'w_q': mhattn.w_q, 
    'w_k': mhattn.w_k,
    'w_v': mhattn.w_v,
    'w_o': mhattn.w_o,
    # FFN
    'ffn_w1': ffn.W1,
    'ffn_b1': ffn.b1,
    'ffn_w2': ffn.W2,
    'ffn_b2': ffn.b2,
    # Layer Norms
    'ln1_g': lnorm.gamma,
    'ln1_b': lnorm.beta,
    'ln2_g': lnorm2.gamma,
    'ln2_b': lnorm2.beta,
    # Final Projection
    'proj_w': proj.W,
    'proj_b': proj.b
}
# Save as a single compressed file
np.savez("quarkus_expert.npz", **state_dict)


