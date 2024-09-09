This is a cheatsheet comparing BERT, GPT3 and LLAMA3

|     | BERT | GPT3 | LLAMA3 |
| --- | --- | --- | --- |
| tokenizer | wordpiece | BPE | TikToken |
| sequence_length | 512 | 2048 | 8192 | 
| positional encoding | absolute | absolute | ROPE |
| architecture | encoder-only | decoder-only | decoder-only |
| activation layer | GELU | GELU | SwiGLU |
| normalization | post layernorm | pre layernorm | pre RMSNorm |
| attention acceleration | None | None | Group Query Attention + KV Cache |
| # layers | 12 layers (12 heads, 768 emb_dim) | 96 layers (96 heads, 128 emb_dim) | 32 layers (32 heads, 128 emb_dim) |
| training dataset | 3.3B words | 450GB | 15T token |
| # params | 110M | 175B | 8B/70B |