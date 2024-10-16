## Transformer Basics

1. Equation for scaled dot self-attention:

$$
\begin{aligned}
\{Self-attention}(X) &= &Softmax(\frac{Q * K^T}{sqrt(D^{model})})V \\
Q &= &X * W^Q\\
K &= &X * W^K\\
V &= &X * W^V
\end{aligned}
$$

where $X\in \mathbb{R}^{N*D^{model}}$, $W^Q\in \mathbb{R}^{D^{model}*D^{model}}$, $W^K\in \mathbb{R}^{D^{model}*D^{model}}$, $W^V\in \mathbb{R}^{D^{model}*D^{model}}$

2. Why the attention score needs to be scaled?

Because Softmax tends to squeeze the gradients of very large values into 0, so scaling the values down can help with maintaining stable gradients calculation.

3. What are the primary positional encoding methods? The corresponding equations?
* sinusoidal positional encoding (BERT, GPT) to input embedding:
  
  $$
  \begin{aligned}
  PE(n, 2*k) &= &\sin(\frac{n}{10000^{2*k/d^{model}}}) \\
  PE(n, 2*k+1) &= &\cos(\frac{n}{10000^{2*k/d^{model}}})
  \end{aligned}
  $$
  
* relative positional encoding (T5):
* learnable positional encoding (ViT):
* rotary positional encoding (Llama) to key/query:

  $$
  \begin{aligned}
  ROPE(x_n, n) = \left\(cos\theta_1,sin\theta_1,...\\
  -sin\theta_1,cos\theta_1,...\\
  ..., cos\theta_d, sin\theta_d,\\
  ..., -sin\theta_d, cos\theta_d
  \)\right
  \end{aligned}
  $$

  4. What's the difference between encoder and decoder?
 For transformers, the decoder uses cross-attention which takes query/key from the encoder

  5. What is layer normalization?
     To differntiate from the batch normalization, which normalize each feature point individually within each batch; LN normalized over each hidden dimension so it doesn't require input to be of the same lengths

4. 
