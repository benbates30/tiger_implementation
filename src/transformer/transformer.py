"""
Notes from paper:

- Their implementation uses T5X -> this uses PyTorch
- Bidirectional encoder + traditional decoder

Encoder:

- 4 layers

Decoder:

- 4 layers
- 6 self attention heads
- Dimension of 64 in each heads i.e. 384 total

- MLP dimension of 1024
- Input dim of 128
- 0.1 Dropout
- ~13 million total paramaters

- 1024 tokens (256codebook size * 4 codebooks) for semantic IDs + 2000 user ID tokens using hashing trick

Training:

- 200k steps on Beauty and Sports and Outdoors, 100k on Toys and Games
- Batch size of 256
- Learning rate of 0.01 for first 10k steps, inverse square root decay schedule from there

TODO: Add configs for each model

"""



import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

import math


class LayerNorm(nn.Module):
    def __init__(self,
                ndim: int,
                bias: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input: Tensor,) -> Tensor:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)



class MLP(nn.Module):

    def __init__(self,
                embed_dim: int,
                ff_dim: int,
                dropout: float) -> None:
        super().__init__()

        self.net = nn.Sequential(
                nn.Linear(embed_dim, ff_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, embed_dim),
            )
        self.t5_layer_norm = LayerNorm(embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    
    def forward(self,
            xs: Tensor) -> Tensor:
        xs_norm = self.t5_layer_norm(xs)
        ys = self.net(xs)
        output = xs + self.dropout(ys)
        return output


class SelfAttention(nn.Module):

    def __init__(self,
                input_dim: int,
                num_heads: int,
                dim_head: int,
                dropout: float,
                ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.dim_head = dim_head
        self.inner_dim = dim_head * num_heads


        self.key = nn.Linear(input_dim, self.inner_dim, bias=False)
        self.query = nn.Linear(input_dim, self.inner_dim, bias=False)
        self.value = nn.Linear(input_dim, self.inner_dim, bias=False)
        self.proj_out = nn.Linear(self.inner_dim, input_dim)




        # self.rel_pos_bias = 5 # TODO: rel posiiton bias

        self.dropout = nn.Dropout(dropout)

        

    def forward(self, x: Tensor, mask=None) -> Tensor:
        # mask decides how to mask the attention

        B, T, C = x.size()

        def _reshape(xs):
            return xs.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)

        def _unshape(self, xs):
            return xs.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        q = _reshape(self.query(x))
        k = _reshape(self.key(x))
        v = _reshape(self.value(x))


        q = q * self.dim_head**-0.5

        score = torch.einsum('b h i d, b h j d -> b h i j', q, k)

        score = self.rel_pos_bias(score)

        mask_value = -torch.finfo(score.dtype).max

        if mask is not None:
            score = score.masked_fill_(~mask, mask_value)


        attn = F.softmax(score, dim=-1)
        attn = self.dropout(attn)

        # aggregate

        y = attn @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = _reshape(y) # re-assemble all head outputs side by side

        return self.proj_out(y) # project back to output embedding dim





class CrossAttention(nn.Module):

    def __init__(self,
                input_dim: int,
                num_heads: int,
                dim_head: int,
                dropout: float,
                context_dim: int = None,
                ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.dim_head = dim_head
        self.inner_dim = dim_head * num_heads

        if context_dim is None:
            context_dim = input_dim

        self.key = nn.Linear(input_dim, self.inner_dim, bias=False)
        self.query = nn.Linear(input_dim, self.inner_dim, bias=False)
        self.value = nn.Linear(input_dim, self.inner_dim, bias=False)
        self.proj_out = nn.Linear(self.inner_dim, input_dim, )

        self.dropout = nn.Dropout(dropout)
        


    def forward(self, x: Tensor, context: Tensor, mask=None, context_mask=None) -> Tensor:
        
        B, T, C = x.size()

        if context is None:
            past_key_value_states = x
        else:
            past_key_value_states = context

        def _reshape(xs):
            return xs.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)

        def _unshape(self, xs):
            return xs.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        q = _reshape(self.query(x))
        k = _reshape(self.key(x))
        v = _reshape(self.value(x))


        q = q * self.dim_head**-0.5

        score = torch.einsum('b h i d, b h j d -> b h i j', q, k)

        mask_value = -torch.finfo(score.dtype).max

        if mask is not None:
            score = score.masked_fill_(~mask, mask_value)

        if context_mask is not None:
            score = score.masked_fill_(~context_mask[:, None, :], mask_value)

        attn = F.softmax(score, dim=-1)
        attn = self.dropout(attn)

        # aggregate

        y = attn @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = _reshape(y) # re-assemble all head outputs side by side

        return self.proj_out(y) # project back to output embedding dim


class EncoderBlock(nn.Module):
    def __init__(self,
                vocab_size: int,
                embed_dim: int,
                num_heads: int,
                dim_head: int,
                mlp_dim: int,
                dropout: float,
                ) -> None:
        super().__init__()
        self.ln1 = LayerNorm(embed_dim, bias=False)
        self.attention = SelfAttention(input_dim=embed_dim, num_heads=num_heads, 
                                       dim_head=dim_head, dropout=dropout)
        self.ln2 = LayerNorm(embed_dim, bias=False)
        self.mlp = MLP(embed_dim=embed_dim, ff_dim=mlp_dim, dropout=dropout)

    def forward(self, x: Tensor, mask=None) -> Tensor:
        # residual connections included
        x = x + self.attention(self.ln1(x), mask=mask)
        x = x + self.mlp(self.ln2(x))
        return x

class Encoder(nn.Module):
    """
    Full encoder class used
    """
    def __init__(self,
                vocab_size: int,
                embed_dim: int,
                num_layers: int,
                num_heads: int,
                dim_head: int,
                mlp_dim: int,
                dropout: float,
                ) -> None:
        super().__init__()

        self.tok_embed = nn.Embedding(vocab_size, embed_dim)
        self.output_norm = LayerNorm(embed_dim, bias=False)
        
        self.layers = nn.ModuleList([EncoderBlock(vocab_size=vocab_size,
                                                  embed_dim=embed_dim,
                                                  num_heads=num_heads,
                                                  dim_head=dim_head,
                                                  mlp_dim=mlp_dim,
                                                  dropout=dropout) for _ in range(num_layers)])

    
    def forward(self, x: Tensor, mask=None) -> Tensor:
        x = self.tok_embed(x)
        for encoder_block in self.layers:
            x = encoder_block(x, mask=mask)
        return self.output_norm(x)
    


class DecoderBlock(nn.Module):

    def __init__(self,
                vocab_size: int,
                embed_dim: int,
                num_heads: int,
                dim_head: int,
                mlp_dim: int,
                dropout: float,
                ) -> None:
        super().__init__()
        self.ln1 = LayerNorm(embed_dim, bias=False)
        self.self_attention = SelfAttention(input_dim=embed_dim, num_heads=num_heads, 
                                            dim_head=dim_head, dropout=dropout)
        self.ln2 = LayerNorm(embed_dim, bias=False)
        self.cross_attention = CrossAttention(embed_dim, num_heads, dim_head, dropout, )
        self.ln3 = LayerNorm(embed_dim, bias=False)
        self.mlp = MLP(embed_dim=embed_dim, ff_dim=mlp_dim, dropout=dropout)

    def forward(self, x: Tensor, context:Tensor, mask=None, context_mask=None) -> Tensor:
        # includes residual connections
        x = x + self.self_attention(self.ln1(x), mask=mask)
        x = x + self.cross_attention(self.ln2(x), context=context, mask=mask, context_mask=context_mask)
        x = x + self.mlp(self.ln3(x))
        
        return x

        


class Decoder(nn.Module):
    """
    Full decoder class used
    """

    def __init__(self,
                vocab_size: int,
                embed_dim: int,
                num_layers: int,
                num_heads: int,
                dim_head: int,
                mlp_dim: int,
                dropout: float,
                ) -> None:
        super().__init__()
        
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.output_norm = LayerNorm(embed_dim, bias=False)
        self.layers = nn.ModuleList([DecoderBlock(vocab_size=vocab_size,
                                                  embed_dim=embed_dim,
                                                  num_heads=num_heads,
                                                  dim_head=dim_head,
                                                  mlp_dim=mlp_dim,
                                                  dropout=dropout) for _ in range(num_layers)])
        
        
    def forward(self, x: Tensor, context:Tensor,  mask=None, context_mask=None) -> Tensor:
        x = self.tok_embed(x)
        for decoder_block in self.layers:
            x = decoder_block(x, context=context, mask=mask, context_mask=context_mask)

        return self.output_norm(x)


class T5(nn.Module):
    """
    Full transformer used by the paper

    TODO: Use a config to not have to pass all the hyperparams through the modules of the model
    """

    def __init__(self, vocab_size: int,
                embed_dim: int,
                enc_blocks: int,
                enc_num_heads: int,
                enc_dim_head: int,
                enc_mlp_dim: int,
                dec_blocks: int,
                dec_num_heads: int,
                dec_dim_head: int,
                dec_mlp_dim: int,
                dropout: float = 0.1,
                ) -> None:
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, embed_dim)

        self.encoder = Encoder(vocab_size=vocab_size,
                               embed_dim=embed_dim,
                               num_layers=enc_blocks,
                               num_heads=enc_num_heads,
                               dim_head=enc_dim_head,
                               mlp_dim=enc_mlp_dim,
                               dropout=dropout)
        
        self.decoder = Decoder(vocab_size=vocab_size,
                               embed_dim=embed_dim,
                               num_layers=dec_blocks,
                               num_heads=dec_num_heads,
                               dim_head=dec_dim_head,
                               mlp_dim=dec_mlp_dim,
                               dropout=dropout)
        
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False) # to logits

    def forward(self, src, target, mask=None, context_mask=None) -> Tensor:
        xs = self.tok_embed(src, mask=mask)
        x = self.encoder(target, src, mask=mask, context_mask=context_mask)

        x = self.lm_head(x)
        return x




if __name__ == "__main__":
    
    transformer = T5(1024, 768, 4, 12, 64, 1028, 4, 12, 64, 1028, 0.1)
    pytorch_total_params = sum(p.numel() for p in transformer.parameters())
    print(pytorch_total_params)