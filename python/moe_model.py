import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

class SimpleAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        S, D = x.shape
        q = self.q_proj(x).view(S, self.num_heads, self.head_dim).transpose(0, 1) # [H, S, head_dim]
        k = self.k_proj(x).view(S, self.num_heads, self.head_dim).transpose(0, 1) # [H, S, head_dim]
        v = self.v_proj(x).view(S, self.num_heads, self.head_dim).transpose(0, 1) # [H, S, head_dim]
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5) # [H, S, S]
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v) # [H, S, head_dim]
        
        context = context.transpose(0, 1).contiguous().view(S, D) # [S, D]
        output = self.o_proj(context)
        return output

class SimpleMoeMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_experts, num_experts_per_tok):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.ModuleDict({
                "gate_proj": nn.Linear(hidden_size, intermediate_size, bias=False),
                "up_proj": nn.Linear(hidden_size, intermediate_size, bias=False),
                "down_proj": nn.Linear(intermediate_size, hidden_size, bias=False)
            }) for _ in range(num_experts)
        ])

    def forward(self, x):
        orig_shape = x.shape
        x_flat = x.view(-1, orig_shape[-1]) # [N, hidden_size]
        
        # Router logits
        router_logits = self.gate(x_flat) # [N, num_experts]
        
        # Top-K
        routing_weights = torch.softmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)
        
        # Normalize topk weights
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        out_flat = torch.zeros_like(x_flat)
        expert_outputs_accum = torch.zeros_like(x_flat)
        
        for i in range(x_flat.shape[0]):
            token_state = x_flat[i]
            for r in range(self.num_experts_per_tok):
                exp_idx = topk_indices[i, r].item()
                weight = topk_weights[i, r]
                
                expert = self.experts[exp_idx]
                w1 = expert["gate_proj"](token_state)
                w3 = expert["up_proj"](token_state)
                silu_val = torch.sigmoid(w1) * w1
                ffn_val = expert["down_proj"](silu_val * w3)
                
                out_flat[i] += weight * ffn_val
                expert_outputs_accum[i] += ffn_val  # Save unweighted FFN outputs for debug verification
                
        return out_flat.view(orig_shape), router_logits, topk_indices, expert_outputs_accum.view(orig_shape)

class SimpleMoeLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_heads, num_experts, num_experts_per_tok):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size)
        self.self_attn = SimpleAttention(hidden_size, num_heads)
        self.post_attention_layernorm = RMSNorm(hidden_size)
        self.mlp = SimpleMoeMLP(hidden_size, intermediate_size, num_experts, num_experts_per_tok)

    def forward(self, x):
        # Attention
        norm_x = self.input_layernorm(x)
        attn_out = self.self_attn(norm_x)
        x = x + attn_out
        
        # MLP
        norm_x2 = self.post_attention_layernorm(x)
        mlp_out, router_logits, topk_indices, expert_outputs = self.mlp(norm_x2)
        x = x + mlp_out
        return x, attn_out, router_logits, topk_indices, expert_outputs, mlp_out

class SimpleMoeModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, intermediate_size, num_layers, num_heads, num_experts, num_experts_per_tok):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            SimpleMoeLayer(hidden_size, intermediate_size, num_heads, num_experts, num_experts_per_tok)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids):
        # input_ids: [S]
        hidden_states = self.embed_tokens(input_ids)
        
        layer_dumps = []
        for i, layer in enumerate(self.layers):
            layer_input = hidden_states
            hidden_states, attn_out, router_logits, topk_indices, expert_outputs, mlp_out = layer(hidden_states)
            layer_dumps.append({
                "hidden_state": layer_input,
                "attention_output": attn_out,
                "router_logits": router_logits,
                "topk_indices": topk_indices,
                "expert_outputs": expert_outputs,  # Unweighted expert FFN sum
                "final_output": hidden_states
            })
            
        final_hidden = self.norm(hidden_states)
        logits = self.lm_head(final_hidden)
        
        return logits, layer_dumps
