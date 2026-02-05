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

    def __call__(self, token_ids: np.ndarray) -> np.ndarray:
        """
        token_ids: (batch, time)
        returns: (batch, time, dim)
        """
        return self.weight[token_ids]

class PositionalEmbedding:
    def __init__(self, context_length: int, dim: int):
        self.weight = np.random.randn(context_length, dim) * 0.01

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        x: (batch, time, dim)
        """
        return x + self.weight[np.newaxis, :, :]


class InputEmbedding:
    def __init__(self, vocab_size, context_length, dim):
        self.token_emb = Embedding(vocab_size, dim)
        self.pos_emb = PositionalEmbedding(context_length, dim)

    def __call__(self, token_ids):
        x = self.token_emb(token_ids)
        x = self.pos_emb(x)
        return x


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

        return out, weights


class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.eps = eps

    def forward(self, x):
        # x: (B, T, D)
        mean = x.mean(axis=-1, keepdims=True)   # (B, T, 1)
        var = x.var(axis=-1, keepdims=True)     # (B, T, 1)

        x_hat = (x - mean) / np.sqrt(var + self.eps)
        out = self.gamma * x_hat + self.beta
        
        return out


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

    
    
