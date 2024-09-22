
        



class multihead_attention(nn.torch):
    def __init__(self, seq_len, num_heads, d_model, mask=None):
        super.__init__(self)
        
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.d_model = d_model
        
        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)
        
        self.linear = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax()
        
        self.rope = rotary_position(seq_len, d_model)    
            
    def split_head(self, x, num_heads):
        # input x: (batch, seq_len, d_model)
        # output x: (batch, num_heads, seq_len, d_model // num_heads)
        batch, seq_len, d_model = x.size()
        x = x.view(batch, seq_len, num_heads, d_model).transpose(1, 2)
        
        return x
    
    def concat_head(self, x):
        # input x: (batch, num_heads, seq_len, d_model // num_heads)
        # output x: (batch, seq_len, d_model)
        batch, num_heads, seq_len, d_head = x.size()
        x = x.transpose(1, 2).view(batch, seq_len, num_heads*d_head)
        
        return x
    
    def self_attention(self, q, k, v, mask=None):
        attention = torch.einsum("ij,kj->ik", q, k)
        attention = self.softmax(attention/torch.sqrt(self.d_model))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        output = torch.einsum("ij,jk->ik", attention, v)
        return output, attention
        
    def forward(self, q, k, v):
        # q: (batch, seq_len, d_model)
        # k: (batch, seq_len, d_model)
        # v: (batch, seq_len, d_model)
        
        q = self.rope(self.WQ(q))
        k = self.rope(self.WK(k))
        v = self.rope(self.WV(v))
        
        q = self.split_head(q, self.num_heads)
        k = self.split_head(k, self.num_heads)
        v = self.split_head(v, self.num_heads)
        
        output, attention = self.dot_attention(q, k, v)
        
        output = self.concat_head(output)
        
        output = self.linear(output)
        
        return output
    
class group_query_attention(nn.torch):
    def __init__(self, seq_len, query_heads, kv_heads, d_model, mask=None):
        """
        MHA:
        
        | (q, q, ..., q)  | (q, q, ..., q) | ... | (q, q, ..., q) | -> query_dim = num_heads * head_dim
        | (k, k, ..., k)  | (k, k, ..., k) | ... | (k, k, ..., k) | ->   key_heads = num_heads * head_dim
        | (v, v, ..., v)  | (v, v, ..., v) | ... | (v, v, ..., v) | -> value_heads = num_heads * head_dim
        
        =====================================================================
        GQA:
        
        | (q, q, ..., q) (q, q, ..., q)...  |  ... | (q, q, ..., q) (q, q, ..., q)|    ->    query_dim = query_heads * head_dim
        |       (k/v, k/v, ..., k/v)       |   ... |       (k/v, k/v, ..., k/v)      | ->    kv_dim = kv_heads * head_dim
                                                                                             group_nums = query_heads / kv_heads
        
        """
        super.__init__(self)
        
        self.seq_len = seq_len
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.d_model = d_model
        self.group_nums = self.query_heads / self.kv_heads
        self.head_dim = self.d_model / self.query_heads   
    
        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, self.head_dim * self.kv_heads)
        self.WV = nn.Linear(d_model, self.head_dim * self.kv_heads)
        
        self.linear = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax()
        
        self.rope = rotary_position(seq_len, d_model)    
            
    def split_group_head(self, x, group_nums, head_dim, type="q"):
        # input q, k, v: (batch, seq_len, x_dim)
        # output q: (batch, group_nums, group_dim, seq_len, head_dim)
        # output k: (batch, group_nums, seq_len, head_dim)
        # output v: (batch, group_nums, seq_len, head_dim)
        
        batch, seq_len, x_dim = x.size()
        if type == "q":
            group_dim = int(x_dim / head_dim / group_nums)
            x = x.view(batch, seq_len, group_nums, group_dim, head_dim)
            x = torch.permute(x, [0, 2, 3, 1, 4])
        else:
            x = x.view(batch, seq_len, group_nums, head_dim).transpose(1, 2)
        
        return x
    
    def concat_head(self, x):
        # input x: (batch, num_heads, seq_len, d_model // num_heads)
        # output x: (batch, seq_len, d_model)
        batch, num_heads, seq_len, d_head = x.size()
        x = x.transpose(1, 2).view(batch, seq_len, num_heads*d_head)
        
        return x
    
    def group_dot_attention(self, q, k, v):
        # q - (batch, group_nums, group_dim, seq_len, head_dim)
        # k - (batch, group_nums, seq_len, head_dim)
        # v - (batch, group_nums, seq_len, head_dim)
        
        attention = torch.einsum("bgdsh,bgsh->bgdsh", q, k)
        attention = self.softmax(attention/torch.sqrt(self.d_model))
        output = torch.einsum("bgdsh,bgsh->bgdsh", attention, v)
        return output, attention
        
    def forward(self, q, k, v):
        # q: (batch, seq_len, d_model)
        # k: (batch, seq_len, kv_dim)
        # v: (batch, seq_len, kv_dim)
        
        q = self.rope(self.WQ(q))
        k = self.rope(self.WK(k))
        v = self.rope(self.WV(v))
        
        q = self.split_group_head(q, self.group_nums, self.head_dim, "q")
        k = self.split_group_head(k, self.group_nums, self.head_dim, "k")
        v = self.split_group_head(v, self.group_nums, self.head_dim, "v")
        
        # q - (batch, group_nums, group_dim, seq_len, head_dim)
        # k - (batch, group_nums, seq_len, head_dim)
        # v - (batch, group_nums, seq_len, head_dim)
        
        output, attention = self.group_dot_attention(q, k, v)
        
        output = self.concat_head(output)
        
        output = self.linear(output)
        
        return output
        
        
        