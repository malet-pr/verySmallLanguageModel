from dataclasses import dataclass
import numpy as np

@dataclass
class Vocabulary:
    stoi: dict  # string (char) -> int
    itos: dict  # int -> string
    size: int


def build_vocab(text: str) -> Vocabulary:
    """
    Build a character-level vocabulary from text.
    """
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return Vocabulary(stoi=stoi, itos=itos, size=len(chars))

def encode(text: str, vocab: Vocabulary) -> np.ndarray:
    """
    Encode text into a numpy array of integer token IDs.
    """
    return np.array([vocab.stoi[ch] for ch in text], dtype=np.int64)


def decode(tokens: np.ndarray, vocab: Vocabulary) -> str:
    """
    Decode a sequence of token IDs back into text.
    """
    return "".join(vocab.itos[int(t)] for t in tokens)


def make_batch(data: np.ndarray, context_length: int, batch_size: int):
    """
    Sample a batch of (input, target) sequences from the data.
    """
    max_start = len(data) - context_length - 1
    starts = np.random.randint(0, max_start, size=batch_size)

    x = np.zeros((batch_size, context_length), dtype=np.int64)
    y = np.zeros((batch_size, context_length), dtype=np.int64)

    for i, start in enumerate(starts):
        chunk = data[start : start + context_length + 1]
        x[i] = chunk[:-1]
        y[i] = chunk[1:]

    return x, y

class Embedding:
    def __init__(self, vocab_size: int, dim: int):
        self.weight = np.random.randn(vocab_size, dim) * 0.01

    def forward(self, x_indices):
        self.x_indices = x_indices 
        return self.weight[x_indices]
    
    def backward(self, dx):
        self.dW = np.zeros_like(self.weight)
        # self.x_indices must be saved in Embedding.forward
        np.add.at(self.dW, self.x_indices, dx)

class PositionalEmbedding:
    def __init__(self, context_length: int, dim: int):
        self.weight = np.random.randn(context_length, dim) * 0.01

    def __call__(self, x: np.ndarray) -> np.ndarray:
        T = x.shape[1] 
        return x + self.weight[:T, :]
    
    def backward(self, dx):
        # We sum over the batch because the same position 
        # weight is used for every sequence in the batch
        self.dW = np.sum(dx, axis=0)


class InputEmbedding:
    def __init__(self, vocab_size, context_length, dim):
        self.token_emb = Embedding(vocab_size, dim)
        self.pos_emb = PositionalEmbedding(context_length, dim)

    def __call__(self, token_ids):
        x = self.token_emb.forward(token_ids)
        x = self.pos_emb(x)
        return x
    def backward(self, dx):
        # 1. Pass the gradient to the positional embeddings
        self.pos_emb.backward(dx)
        # 2. Pass the gradient to the token embeddings
        self.token_emb.backward(dx)
            

def causal_mask(T):
    mask = np.triu(np.ones((T, T)), k=1)
    mask = mask * -1e9
    return mask


class SelfAttention:
    def __init__(self, dim):
        self.dim = dim

        # Projection matrices (learned)
        self.W_q = np.random.randn(dim, dim) * 0.02
        self.W_k = np.random.randn(dim, dim) * 0.02
        self.W_v = np.random.randn(dim, dim) * 0.02

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape

        # Project to Q, K, V
        Q = x @ self.W_q          # (B, T, D)
        K = x @ self.W_k          # (B, T, D)
        V = x @ self.W_v          # (B, T, D)

        # Attention scores
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(D)  # (B, T, T)

        # Causal mask (no looking ahead)
        mask = np.triu(np.ones((T, T)), k=1)
        scores = scores - 1e9 * mask

        # Softmax (attention weights)
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        weights = weights / weights.sum(axis=-1, keepdims=True)

        # Weighted sum of values
        out = weights @ V          # (B, T, D)
        
        self.weights = weights
        self.V = V
        self.K = K
        self.Q = Q
        self.x = x

        return out, weights
    
    def backward(self, dout):
        # dout: (B, T, D)
        B, T, D = dout.shape
        
        # We need these from the forward pass (make sure to cache them!)
        # self.weights: (B, T, T)
        # self.V: (B, T, D)
        # self.Q, self.K: (B, T, D)
        
        # 1. Gradient wrt V and weights
        # out = weights @ V
        self.dV = self.weights.transpose(0, 2, 1) @ dout  # (B, T, D)
        dweights = dout @ self.V.transpose(0, 2, 1)       # (B, T, T)

        # 2. Gradient wrt Softmax scores
        # This is the 'stable softmax' backward
        dscores = self.weights * (dweights - np.sum(dweights * self.weights, axis=-1, keepdims=True))

        # 3. Masking (Future tokens don't contribute to the gradient)
        mask = np.triu(np.ones((T, T)), k=1)
        dscores[:, mask == 1] = 0
        
        # 4. Scaling back
        dscores /= np.sqrt(D)
        
        # 5. Gradient wrt Q and K
        # scores = Q @ K.T
        dQ = dscores @ self.K                       # (B, T, D)
        dK = dscores.transpose(0, 2, 1) @ self.Q    # (B, T, D)

        # 6. Gradient wrt Projection Weights (W_q, W_k, W_v)
        x_flat = self.x.reshape(-1, D)
        self.dW_q = x_flat.T @ dQ.reshape(-1, D)
        self.dW_k = x_flat.T @ dK.reshape(-1, D)
        self.dW_v = x_flat.T @ self.dV.reshape(-1, D)

        # 7. Final dx to pass back
        dx = (dQ @ self.W_q.T) + (dK @ self.W_k.T) + (self.dV @ self.W_v.T)
        dx = dx.reshape(B, T, D)
        return dx,None


class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.eps = eps

    def forward(self, x):
        # x: (B, T, D)
        self.x = x
        mean = x.mean(axis=-1, keepdims=True)   # (B, T, 1)
        var = x.var(axis=-1, keepdims=True)     # (B, T, 1)
        
        self.std = np.sqrt(var + self.eps)
        self.norm_x = (x - mean) / self.std

        return self.gamma * self.norm_x + self.beta
    
    def backward(self, dout):
        # dout: (B, T, D)
        B, T, D = dout.shape
        
        # 1. Gradients for the parameters
        # We sum over Batch and Time because gamma/beta are shared 
        # across the whole sequence (shape: 1, D)
        self.dgamma = np.sum(dout * self.norm_x, axis=(0, 1), keepdims=True)
        self.dbeta = np.sum(dout, axis=(0, 1), keepdims=True)

        # 2. Gradient for the input (dx)
        # This is the "mean-field" derivative. It's a bit of a beast:
        dx_norm = dout * self.gamma
        dx = (1. / (D * self.std)) * (
            D * dx_norm - 
            np.sum(dx_norm, axis=-1, keepdims=True) - 
            self.norm_x * np.sum(dx_norm * self.norm_x, axis=-1, keepdims=True)
        )
        return dx


class OutputProjection:
    def __init__(self, dim, vocab_size):
        self.W = np.random.randn(dim, vocab_size) * 0.02
        self.b = np.zeros(vocab_size)

    def forward(self, x):
        self.x = x  # cache for backward
        return x @ self.W + self.b

    def backward(self, dlogits):
        # dlogits: (B, T, V)
        B, T, V = dlogits.shape
        D = self.W.shape[0]

        x2d = self.x.reshape(B*T, D)
        dlog2d = dlogits.reshape(B*T, V)

        self.dW = x2d.T @ dlog2d
        self.db = dlog2d.sum(axis=0)

        dx = (dlog2d @ self.W.T).reshape(B, T, D)
        return dx


def cross_entropy_loss(logits, targets):
    # logits: (B, T, V)
    # targets: (B, T)

    # numerical stability
    logits = logits - logits.max(axis=-1, keepdims=True)

    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    B, T = targets.shape
    idx_b = np.arange(B)[:, None]
    idx_t = np.arange(T)[None, :]

    correct_probs = probs[idx_b, idx_t, targets]
    loss = -np.log(correct_probs).mean()

    return loss, probs

def cross_entropy_backward(probs, targets):
    B, T, V = probs.shape

    dlogits = probs.copy()
    idx_b = np.arange(B)[:, None]
    idx_t = np.arange(T)[None, :]
    dlogits[idx_b, idx_t, targets] -= 1
    dlogits /= (B * T)

    return dlogits


class FFN:
    def __init__(self, dim, hidden_dim):
        self.W1 = np.random.randn(dim, hidden_dim) * 0.02
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, dim) * 0.02
        self.b2 = np.zeros(dim)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # x: (B, T, D)
        self.x = x

        self.h = x @ self.W1 + self.b1        # (B,T,4D)
        self.h_relu = np.maximum(0, self.h)  # ReLU
        out = self.h_relu @ self.W2 + self.b2 # (B,T,D)

        return out

    def backward(self, dout):
        # dout: (B, T, D)
        B, T, D = dout.shape
        BT = B * T

        # ---- second linear ----
        dout2d = dout.reshape(BT, D)
        h2d = self.h_relu.reshape(BT, self.hidden_dim)

        self.dW2 = h2d.T @ dout2d
        self.db2 = dout2d.sum(axis=0)
        dh_relu = (dout2d @ self.W2.T).reshape(B, T, self.hidden_dim)

        # ---- ReLU ----
        dh = dh_relu * (self.h > 0)

        # ---- first linear ----
        x2d = self.x.reshape(BT, D)
        dh2d = dh.reshape(BT, self.hidden_dim)

        self.dW1 = x2d.T @ dh2d
        self.db1 = dh2d.sum(axis=0)
        dx = (dh2d @ self.W1.T).reshape(B, T, D)

        return dx

def grad_check(model_layer, param_name, input_data, epsilon=1e-6):
    """
    Checks the analytic gradient against the numerical gradient.
    """
    # 1. Get the analytic gradient
    # We run a forward and backward pass first
    out = model_layer.forward(input_data)
    # Assuming we are testing a layer that returns (out, weights) like Attention
    if isinstance(out, tuple): out = out[0]
    
    # Fake a 'dout' (gradient from above)
    dout = np.random.randn(*out.shape)
    
    # Run backward to get the dW we want to check
    model_layer.backward(dout)
    analytic_grad = getattr(model_layer, f"d{param_name}")
    
    # 2. Calculate Numerical Gradient
    param = getattr(model_layer, param_name)
    num_grad = np.zeros_like(param)
    
    # We iterate through every single element in the weight matrix
    # (Warning: This is slow, so we only do it for small matrices or samples!)
    it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        old_val = param[ix]
        
        # f(x + h)
        param[ix] = old_val + epsilon
        out_plus = model_layer.forward(input_data)
        if isinstance(out_plus, tuple): out_plus = out_plus[0]
        loss_plus = np.sum(out_plus * dout) # Scalar projection
        
        # f(x - h)
        param[ix] = old_val - epsilon
        out_minus = model_layer.forward(input_data)
        if isinstance(out_minus, tuple): out_minus = out_minus[0]
        loss_minus = np.sum(out_minus * dout)
        
        # (f(x+h) - f(x-h)) / 2h
        num_grad[ix] = (loss_plus - loss_minus) / (2 * epsilon)
        
        param[ix] = old_val # Reset to original
        it.iternext()
        
    # 3. Compare
    rel_error = np.linalg.norm(analytic_grad - num_grad) / (np.linalg.norm(analytic_grad + num_grad) + 1e-10)
    
    print(f"Checking gradient for {param_name}...")
    if rel_error < 1e-7:
        print(f"✅ PASSED! Relative Error: {rel_error:.2e}")
    else:
        print(f"❌ FAILED! Relative Error: {rel_error:.2e}")
        
