
import numpy as np


def softmax(x):
    # Numerically stable row-wise softmax
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# Four tokens, each represented by 4 features
X = np.array([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 2.0, 0.0, 2.0],
    [1.0, 1.0, 1.0, 1.0],
    [2.0, 0.0, 2.0, 0.0],
])

np.random.seed(42)

d_model = 4
d_k = 2

# Single-head self-attention
Wq = np.random.randn(d_model, d_k)
Wk = np.random.randn(d_model, d_k)
Wv = np.random.randn(d_model, d_k)

Q = X @ Wq
K = X @ Wk
V = X @ Wv

scores = Q @ K.T / np.sqrt(d_k)
attention_weights = softmax(scores)
output = attention_weights @ V

print("Single-head attention matrix (4x4):")
print(np.round(attention_weights, 4))

print("\nRow sums:")
print(attention_weights.sum(axis=1))

assert attention_weights.shape == (4, 4)
assert np.allclose(attention_weights.sum(axis=1), 1.0)

print("\nSingle-head attention checks passed!")


# Two-head attention
num_heads = 2
head_dim = 2
heads = []

for head in range(num_heads):
    Wq_h = np.random.randn(d_model, head_dim)
    Wk_h = np.random.randn(d_model, head_dim)
    Wv_h = np.random.randn(d_model, head_dim)

    Q_h = X @ Wq_h
    K_h = X @ Wk_h
    V_h = X @ Wv_h

    scores_h = Q_h @ K_h.T / np.sqrt(head_dim)
    weights_h = softmax(scores_h)
    head_output = weights_h @ V_h

    assert np.allclose(weights_h.sum(axis=1), 1.0)
    heads.append(head_output)

# Concatenate both heads
concatenated = np.concatenate(heads, axis=1)

# Output projection: concatenated dimension 4 -> model dimension 4
W_o = np.random.randn(num_heads * head_dim, d_model)
multi_head_output = concatenated @ W_o

print("\nConcatenated heads shape:", concatenated.shape)
print("Output projection shape:", multi_head_output.shape)
print("\nMulti-head output:")
print(np.round(multi_head_output, 4))

assert multi_head_output.shape == (4, d_model)

print("\nMulti-head attention checks passed!")
