"""numpy_lstm.py — NumPy-only LSTM autoencoder used as fallback when TensorFlow
is unavailable. Produces results equivalent (within seed noise) to the Keras
implementation in Solar_Entropy_Pipeline.ipynb.
"""
import numpy as np
SEED = 42

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

class LSTMCell:
    """Single LSTM cell, input_dim x hidden_dim."""
    def __init__(self, input_dim, hidden_dim, seed=0):
        rng = np.random.default_rng(seed)
        k = 1.0 / np.sqrt(hidden_dim)
        self.H = hidden_dim
        # Combined weights: [W_f, W_i, W_c, W_o]
        self.Wx = rng.uniform(-k, k, (input_dim, 4 * hidden_dim)).astype(np.float32)
        self.Wh = rng.uniform(-k, k, (hidden_dim, 4 * hidden_dim)).astype(np.float32)
        self.b  = np.zeros(4 * hidden_dim, dtype=np.float32)
        # Forget-gate bias bump (Jozefowicz et al. 2015)
        self.b[hidden_dim:2*hidden_dim] = 1.0

    def forward_seq(self, X):
        """X: (T, input_dim). Returns hs (T, H), cs (T, H)."""
        T = X.shape[0]
        H = self.H
        h = np.zeros(H, dtype=np.float32)
        c = np.zeros(H, dtype=np.float32)
        hs = np.zeros((T, H), dtype=np.float32)
        cs = np.zeros((T, H), dtype=np.float32)
        caches = []
        for t in range(T):
            z = X[t] @ self.Wx + h @ self.Wh + self.b
            f = sigmoid(z[:H])
            i = sigmoid(z[H:2*H])
            g = np.tanh(z[2*H:3*H])
            o = sigmoid(z[3*H:])
            c = f * c + i * g
            h = o * np.tanh(c)
            hs[t] = h
            cs[t] = c
            caches.append((X[t], f, i, g, o, c, h))
        return hs, cs, caches

    def backward_seq(self, caches, dhs, dcs_last=None):
        """Backprop through time. dhs: (T, H) gradient wrt h at every t."""
        T = len(caches)
        H = self.H
        dWx = np.zeros_like(self.Wx)
        dWh = np.zeros_like(self.Wh)
        db  = np.zeros_like(self.b)
        dh_next = np.zeros(H, dtype=np.float32)
        dc_next = np.zeros(H, dtype=np.float32)
        dX = np.zeros((T, self.Wx.shape[0]), dtype=np.float32)
        h_prev_list = [np.zeros(H, dtype=np.float32)] + [caches[t][6] for t in range(T-1)]
        c_prev_list = [np.zeros(H, dtype=np.float32)] + [caches[t][5] for t in range(T-1)]
        for t in reversed(range(T)):
            x_t, f, i, g, o, c, h = caches[t]
            dh = dhs[t] + dh_next
            do = dh * np.tanh(c)
            dc = dh * o * (1 - np.tanh(c) ** 2) + dc_next
            df = dc * c_prev_list[t]
            di = dc * g
            dg = dc * i
            dc_prev = dc * f
            # Pre-activation gradients
            dz_f = df * f * (1 - f)
            dz_i = di * i * (1 - i)
            dz_g = dg * (1 - g ** 2)
            dz_o = do * o * (1 - o)
            dz = np.concatenate([dz_f, dz_i, dz_g, dz_o])
            dWx += np.outer(x_t, dz)
            dWh += np.outer(h_prev_list[t], dz)
            db  += dz
            dX[t] = dz @ self.Wx.T
            dh_next = dz @ self.Wh.T
            dc_next = dc_prev
        return dX, dWx, dWh, db


class LSTMAutoencoder:
    """
    Sequence-to-sequence autoencoder:
      encoder LSTM (input=1, hidden=latent) takes W=64 samples -> final h = z (latent)
      decoder takes z repeated W times, LSTM (input=latent, hidden=latent) -> Dense(1) recon
    Trained with Adam, MSE loss.
    """
    def __init__(self, window=64, latent=16, seed=SEED):
        self.W = window
        self.L = latent
        self.enc = LSTMCell(1, latent, seed=seed)
        self.dec = LSTMCell(latent, latent, seed=seed + 1)
        rng = np.random.default_rng(seed + 2)
        k = 1.0 / np.sqrt(latent)
        self.Wd = rng.uniform(-k, k, (latent, 1)).astype(np.float32)
        self.bd = np.zeros(1, dtype=np.float32)
        # Adam state
        self.params = ['enc.Wx','enc.Wh','enc.b','dec.Wx','dec.Wh','dec.b','Wd','bd']
        self._init_adam()

    def _init_adam(self):
        self.m = {}
        self.v = {}
        for p in self.params:
            ref = self._get(p)
            self.m[p] = np.zeros_like(ref)
            self.v[p] = np.zeros_like(ref)
        self.t = 0

    def _get(self, name):
        obj = self
        for part in name.split('.'):
            obj = getattr(obj, part)
        return obj

    def _set(self, name, val):
        parts = name.split('.')
        obj = self
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], val)

    def encode(self, x_seq):
        """x_seq: (W,) -> latent (L,)."""
        X = x_seq.reshape(-1, 1).astype(np.float32)
        hs, cs, _ = self.enc.forward_seq(X)
        return hs[-1]  # final hidden state = latent

    def encode_batch(self, X_batch):
        """X_batch: (N, W) -> latents (N, L)."""
        N = X_batch.shape[0]
        Z = np.zeros((N, self.L), dtype=np.float32)
        for i in range(N):
            Z[i] = self.encode(X_batch[i])
        return Z

    def forward(self, x_seq):
        """Full forward pass. Returns recon (W,), plus caches for backward."""
        X = x_seq.reshape(-1, 1).astype(np.float32)
        enc_hs, enc_cs, enc_caches = self.enc.forward_seq(X)
        z = enc_hs[-1]  # (L,)
        # Decoder input = z repeated W times
        Z_rep = np.tile(z, (self.W, 1))
        dec_hs, dec_cs, dec_caches = self.dec.forward_seq(Z_rep)
        # Dense layer on decoder hidden states
        recon = dec_hs @ self.Wd + self.bd  # (W, 1)
        return recon.flatten(), z, enc_caches, dec_caches, dec_hs, Z_rep

    def backward(self, x_seq, recon, z, enc_caches, dec_caches, dec_hs):
        """Backprop MSE loss. Returns gradients dict."""
        target = x_seq.astype(np.float32)
        dL_dr = 2.0 * (recon - target) / self.W  # (W,)
        dL_dr = dL_dr.reshape(-1, 1)
        # Dense grads
        dWd = dec_hs.T @ dL_dr
        dbd = dL_dr.sum(axis=0)
        dDec_h = dL_dr @ self.Wd.T  # (W, L)
        # Decoder backward (input was Z_rep)
        dZ_rep, dDec_Wx, dDec_Wh, dDec_b = self.dec.backward_seq(dec_caches, dDec_h)
        # dz comes from sum over time of dZ_rep
        dz = dZ_rep.sum(axis=0)  # (L,)
        # Encoder backward: only final step has grad = dz
        dEnc_h = np.zeros((self.W, self.L), dtype=np.float32)
        dEnc_h[-1] = dz
        dX, dEnc_Wx, dEnc_Wh, dEnc_b = self.enc.backward_seq(enc_caches, dEnc_h)
        return {
            'enc.Wx': dEnc_Wx, 'enc.Wh': dEnc_Wh, 'enc.b': dEnc_b,
            'dec.Wx': dDec_Wx, 'dec.Wh': dDec_Wh, 'dec.b': dDec_b,
            'Wd': dWd, 'bd': dbd,
        }

    def adam_step(self, grads, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        for p in self.params:
            g = grads[p]
            self.m[p] = beta1 * self.m[p] + (1 - beta1) * g
            self.v[p] = beta2 * self.v[p] + (1 - beta2) * (g ** 2)
            m_hat = self.m[p] / (1 - beta1 ** self.t)
            v_hat = self.v[p] / (1 - beta2 ** self.t)
            self._set(p, self._get(p) - lr * m_hat / (np.sqrt(v_hat) + eps))

    def fit(self, X_train, X_val, epochs=10, lr=1e-3, verbose=True):
        history = {'loss': [], 'val_loss': []}
        N = len(X_train)
        for epoch in range(epochs):
            # Shuffle
            idx = np.random.permutation(N)
            losses = []
            for i in idx:
                recon, z, ec, dc, dh, _ = self.forward(X_train[i])
                loss = float(np.mean((recon - X_train[i]) ** 2))
                grads = self.backward(X_train[i], recon, z, ec, dc, dh)
                self.adam_step(grads, lr=lr)
                losses.append(loss)
            val_losses = []
            for i in range(len(X_val)):
                recon, _, _, _, _, _ = self.forward(X_val[i])
                val_losses.append(float(np.mean((recon - X_val[i]) ** 2)))
            history['loss'].append(float(np.mean(losses)))
            history['val_loss'].append(float(np.mean(val_losses)))
            if verbose:
                print(f"  epoch {epoch+1:2d}/{epochs}  loss={history['loss'][-1]:.6f}  val_loss={history['val_loss'][-1]:.6f}")
        return history