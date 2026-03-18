"""
Transformer encoder with learnable causal adjacency matrix and row-attention.
"""
import torch
from torch import nn
from torch.nn import functional as F


class TransformerEncoder(nn.Module):
    def __init__(self,
                 d_model: int,
                 ffn_factor: float,
                 hidden_dim: int,
                 dropout_rate: float,
                 encoder_depth: int = 3,
                 n_head: int = 2,
                 max_seq_len: int = 1024, # 预留足够大的特征维度空间
                 **kwargs) -> None:
        """
        Upgraded FT-Transformer Encoder with Learnable Causal Mask & Row-Attention.
        """
        super().__init__()

        if d_model % n_head != 0:
            divisors = [n for n in range(1, d_model + 1) if d_model % n == 0]
            closest_num_heads = min(divisors, key=lambda x: abs(x - n_head))
            if closest_num_heads != n_head:
                print(f"Adjusting num_heads from {n_head} to {closest_num_heads} (closest valid divisor of {d_model})")
            n_head = closest_num_heads

        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=n_head, 
                dim_feedforward=int(hidden_dim * ffn_factor), 
                dropout=dropout_rate, 
                batch_first=True
            )
            for _ in range(encoder_depth)
        ])

        self.output_dim = d_model

        # --- [核心改造 1] 因果图挖掘：可学习的全局特征邻接矩阵 ---
        # 预设足够大的 shape，forward 时根据实际 Seq 长度截取，保证 Optimizer 不报错
        self.adj_matrix = nn.Parameter(torch.randn(max_seq_len, max_seq_len) * 0.01)

        # --- [核心改造 2] 拓扑信息交换：批次内行注意力 (Row-Attention) ---
        self.row_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head, batch_first=True)
        self.row_norm = nn.LayerNorm(d_model)

    def get_dag_penalty(self, seq_len: int) -> torch.Tensor:
        """
        提供给外部计算 NOTEARS 迹约束 (Trace Penalty) 的接口。
        用法：在 LightningModule 的 training_step 里调用并加上 `loss += lambda * model.encoder.get_dag_penalty(seq_len)`
        """
        adj = self.adj_matrix[:seq_len, :seq_len]
        W = F.softplus(adj)  # 保证权重非负
        W = W - torch.diag_embed(torch.diagonal(W)) # 消除自环
        
        # 计算迹约束: tr(e^{W \circ W}) - d
        M = W * W
        E = torch.matrix_exp(M)
        trace = torch.trace(E)
        return trace - seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, Seq, D = x.shape

        # 1. 生成平滑且可微的因果掩码 (Causal Mask)
        adj = self.adj_matrix[:Seq, :Seq]
        W = F.softplus(adj)
        W = W - torch.diag_embed(torch.diagonal(W))
        
        # 转换为 Transformer 需要的 additive mask: 权重接近 0 的地方阻断注意力 (-1e4)
        src_mask = -1e4 * torch.exp(-W)

        # 2. 列注意力计算 (融入特征间的因果约束)
        for layer in self.encoder_layers:
            # src_mask shape: (Seq, Seq)，会对 batch 中所有样本应用相同的因果图
            x = layer(x, src_mask=src_mask)
            
        cls_token = x[:, 0]

        # 3. 行注意力计算 (融入学生间的社交拓扑)
        # 极度关键：推理(特征提取)时必须关闭，以保证特征向量的确定性！
        if self.training:
            # 将 Batch 维度转换为 Sequence 维度: [B, D] -> [1, B, D]
            row_x = cls_token.unsqueeze(0)
            # 同批次学生互相 Query，形成社区聚集倾向
            attn_out, _ = self.row_attn(row_x, row_x, row_x)
            # 加上残差连接并归一化 -> [B, D]
            cls_token = self.row_norm(cls_token + attn_out.squeeze(0))

        return cls_token
