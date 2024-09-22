import torch.nn as nn
from self_attention import group_query_attention, rmsnorm

class transformer_layer(nn.Module):
    def __init__(self):
        super.__init__(self)
        
        self.rmsnorm = rmsnorm()
        self.gqa = group_query_attention()
        self.ffw = nn.Sequential([nn.Linear(d_model, 2*d_model),
                                  nn.SiLU()
                                  nn.Linear(2*d_model, d_model),
        ])

    def forward(self, x):
        x = self.rmsnorm(x)
        x, _ = self.gqa(x, x, x)
        x = self.ffw(x)
        return x
        
class llama_model(nn.Module):
    def __init__(self):
        super.__init__(self)
        
        self.transformer_layers = nn.Sequential([
            transformer_layer() for idx in range(6)
        ])
        
    def forward(self, x):
        x = self.transformer_layers(x)
        
        return x
        
        
        
            