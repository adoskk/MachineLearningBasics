## Transformer Basics

1. Equation for scaled dot self-attention:

```math
\begin{aligned}
{Self-attention}(X) &= &Softmax\left(\frac{Q * K^T}{\sqrt{D^{model}}}\right)V \\
Q &= &X * W^Q\\
K &= &X * W^K\\
V &= &X * W^V
\end{aligned}
```

where $X\in \mathbb{R}^{N*D^{model}}$, $W^Q\in \mathbb{R}^{D^{model}*D^{model}}$, $W^K\in \mathbb{R}^{D^{model}*D^{model}}$, $W^V\in \mathbb{R}^{D^{model}*D^{model}}$

2. Why the attention score needs to be scaled?

Because Softmax tends to squeeze the gradients of very large values into 0, so scaling the values down can help with maintaining stable gradients calculation.

3. What are the primary positional encoding methods? The corresponding equations?
* sinusoidal positional encoding (BERT, GPT) to input embedding:

```math
\begin{aligned}
PE(n, 2*k) &= &\sin{\frac{n}{10000^{2*k/d^{model}}}} \\
PE(n, 2*k+1) &= &\cos{\frac{n}{10000^{2*k/d^{model}}}}
\end{aligned}
```
  
* relative positional encoding (T5):
* learnable positional encoding (ViT):
* rotary positional encoding (Llama) to key/query:

 ```math
  \begin{aligned}
  ROPE(x_n, n) &= &\left(cos\theta_1,sin\theta_1,...\\
  & &-sin\theta_1,cos\theta_1,...\\
  & &..., cos\theta_d, sin\theta_d,\\
  & &..., -sin\theta_d, cos\theta_d \right)
  \end{aligned}
  ```

  4. What's the difference between encoder and decoder in transformers?
 For transformers, the encoder layer is self-attention layer + FFW layer; the decoder layer is self-attention layer + cross-attention (query/key from the encoder) + FFW. 

  5. What is layer normalization?
     To differntiate from the batch normalization, which normalize each feature point individually within each batch; LN normalized over each hidden dimension so it doesn't require input to be of the same lengths

4. What is the computational cost of the self-attention? $n^2*d^{model}$

## LLM-related

1. What is RAG? Retrieval augmented generation - pass in relevant documents as contextual information to the LLM for generation purposes. Usually, it includes a few steps: convert relevant documents into vector embeddings; the embeddings are passed together with the prompt into the LLM for generation purposes
2. What is LoRA? Is low-rank adaptation for supervised fine-tuning of LLMs. Instead of finetuning the original LLM, LoRA keeps the pretrained Linear/self-attention layer weights unchanged and only train the the multiplication of two low-rank matrices so to reduce computation cost.
3. What is ReLU, GELU, and SwiGLU?

```math

\begin{aligned}
ReLU(x) &= &\left\{
\begin{aligned}
x &x > 0 \\
0 &\text{else}
\end{aligned} \right. \\

GELU(x) &= &0.5*x*[1 + erf(\frac{x}{\sqrt(2)})] \\

SwiGLU(x) &= &x \sigma (\beta*x) + (1 - \sigma(\beta*x)) * (Wx + b)
\end{aligned}
```


4. Waht is RMSNorm? It's an improved version of layer normalization. Instead of subtracting the mean and dividing over std, rmsnorm calculates the l2 norm of the input embedding vector and normalize over it.
5. What are the different normalization techniques of BERT, GPT, and Llama? Why most LLM uses pre-norm than post-norm? BERT uses post-layer normalization, GPT uses pre-layer normalization, and Llama uses Pre-layer RMS normalization. Pre-norm means the normalization is added before the self-attention layer and FFW, which normalizes the input so the training is more stable and the model converges faster.
6. What are the mainstream LLMs? BERT (encoder only), GPT (decoder only), T5 (encoder-decoder), Llama (smaller than GPT3, with RoPE, RMSNorm, KV Cache and MQA), Mistral (RoPE,MOE, KV Cache, MQA, Sliding Window Attention) 
