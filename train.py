import numpy as np
from pathlib import Path
import pickle
from model import *

# Create a config dictionary or Class
config = {
    "vocab_size": None,
    "dim": 128,              
    "context_length": 64,    
    "batch_size": 16,
    "lr": 0.0005,
    "n_heads": 4          
}

iterations = 50000

def main():
    text = Path("data/cortazar.txt").read_text(encoding="utf-8")
    vocab = build_vocab(text)
    data = encode(text, vocab)
    config["vocab_size"] = len(vocab.stoi)
    
    embed = InputEmbedding(vocab_size=vocab.size,context_length=config["context_length"],dim=config["dim"])
    lnorm = LayerNorm(dim=config["dim"])
    attn = SelfAttention(dim=config["dim"])
    lnorm2 = LayerNorm(dim=config["dim"])
    ffn = FFN(dim=config["dim"], hidden_dim=4*config["dim"])
    proj = OutputProjection(dim=config["dim"],vocab_size=vocab.size)

    for i in range(iterations):
        x, y = make_batch(data, context_length=config["context_length"], batch_size=config["batch_size"])
        x_emb = embed(x)
        # A common pattern: Decay down to 10% of the original LR
        min_lr = config['lr'] * 0.1 
        decay_ratio = i / iterations
        current_lr = config['lr'] * (1 - decay_ratio)
        current_lr = max(current_lr, min_lr)

        # forward
        x_norm = lnorm.forward(x_emb)
        attn_out, _ = attn.forward(x_norm)    
        x_res = x_emb + attn_out
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
        dx_attn, _ = attn.backward(dx_res)
        dx_norm1 = lnorm.backward(dx_attn)
        dx_final = dx_norm1 + dx_res
        embed.backward(dx_final)
        
        #update weights
        attn.W_q -= current_lr * attn.dW_q
        attn.W_k -= current_lr * attn.dW_k
        attn.W_v -= current_lr * attn.dW_v
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
     
        if i == 0 or i == iterations-1:
            print(f"Iteration {i}, Loss: {loss:.4f}")
        if i % 100 == 0:
            print("*", end="", flush=True)    

 # After the loop...
    # Create a dictionary of all your trained weights
    state_dict = {
        # Embeddings
        'token_emb': embed.token_emb.weight, 
        'pos_emb': embed.pos_emb.weight,
        # Attention
        'w_q': attn.W_q,
        'w_k': attn.W_k,
        'w_v': attn.W_v,
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
    np.savez("cortazar_expert.npz", **state_dict)
    # Save the vocab
    with open("vocab_cortazar.pkl", "wb") as f:
        pickle.dump({'stoi': vocab.stoi, 'itos': vocab.itos}, f)
 

    my_model = {
        'embed': embed, 'lnorm': lnorm, 'attn': attn,
        'lnorm2': lnorm2, 'ffn': ffn, 'proj': proj,
        'context_length': 32, 'stoi': vocab.stoi, 'itos': vocab.itos
    }
    
    print(generate(my_model, "había una vez ",200))

def generate(model, start_str, gen_length=100, temperature=0.8):
    # Unpack the components from the dictionary
    embed = model['embed']
    lnorm = model['lnorm']
    attn = model['attn']
    lnorm2 = model['lnorm2']
    ffn = model['ffn']
    proj = model['proj']
    context_length = model['context_length']
    stoi = model['stoi']
    itos = model['itos']

    current_context = [stoi[c] for c in start_str if c in stoi]
    result = start_str
    
    for _ in range(gen_length):
        x_input = np.array(current_context[-context_length:]).reshape(1, -1)
        
        # Forward pass
        x = embed(x_input)
        x_norm = lnorm.forward(x)
        attn_out, _ = attn.forward(x_norm)    
        x_res = x_norm + attn_out
        x_ffn_norm = lnorm2.forward(x_res)
        ffn_out = ffn.forward(x_ffn_norm)
        x_out = x_res + ffn_out
        logits = proj.forward(x_out)
        
        last_logits = logits[0, -1, :]
        
        # Softmax + Sampling
        probs = np.exp(last_logits / temperature)
        probs /= np.sum(probs)
        next_id = np.random.choice(len(itos), p=probs)
        result += itos[next_id]
        current_context.append(next_id)
        
    return result

if __name__ == "__main__":
    main()
