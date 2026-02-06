import numpy as np
import pickle
from model import *

# 1. LOAD VOCAB
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)
stoi, itos = vocab['stoi'], vocab['itos']

# 2. LOAD WEIGHTS
weights = np.load("java_expert.npz")

config = {
    "dim": 128,              
    "context_length": 64,    
    "n_heads": 4          
}

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

# 5. PACK FOR GENERATOR
my_model = {
    'embed': embed, 'lnorm': lnorm, 'mhattn': mhattn,
    'lnorm2': lnorm2, 'ffn': ffn, 'proj': proj,
    'context_length': config['context_length'],
    'stoi': stoi, 'itos': itos
}

def generate(model, start_str, gen_length=150, temperature=0.1):
    # Unpack the components from the dictionary
    embed = model['embed']
    lnorm = model['lnorm']
    mhattn = model['mhattn']
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
        x_emb = embed(x_input)
        x_norm = lnorm.forward(x_emb)
        mhattn_out = mhattn.forward(x_norm)   
        x_res = x_emb + mhattn_out
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

# 6. RUN PREDICTION
print(generate(my_model, "public String test() { \nInteger a =", 250, temperature=0.1))