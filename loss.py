# import torch
# from torch import nn
# from torch.nn import functional as F


# class NTXentLoss(nn.Module):
#     def __init__(self, temperature=1.0):
#         """NT-Xent loss for contrastive learning using cosine distance as similarity metric as used in [SimCLR](https://arxiv.org/abs/2002.05709).
#         Implementation adapted from https://theaisummer.com/simclr/#simclr-loss-implementation
#         Args:
#             temperature (float, optional): scaling factor of the similarity metric. Defaults to 1.0.
#         """
#         super().__init__()
#         self.temperature = temperature

#     def forward(self, z_i, z_j):
#         """Compute NT-Xent loss using only anchor and positive batches of samples. Negative samples are the 2*(N-1) samples in the batch
#         Args:
#             z_i (torch.tensor): anchor batch of samples
#             z_j (torch.tensor): positive batch of samples
#         Returns:
#             float: loss
#         """
#         batch_size = z_i.size(0)

#         # Compute similarity between the sample's embedding and its corrupted view
#         z = torch.cat([z_i, z_j], dim=0)
#         similarity = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

#         # Extract positive samples
#         positives = similarity[range(batch_size), range(
#             batch_size, 2 * batch_size)]
#         positives = torch.cat([positives, similarity[range(
#             batch_size, 2 * batch_size), range(batch_size)]], dim=0)

#         # Create mask to exclude self-comparisons
#         mask = torch.ones((2 * batch_size, 2 * batch_size),
#                           dtype=torch.bool, device=z.device)
#         mask.fill_diagonal_(0)

#         # Compute numerator and denominator
#         exp_sim = torch.exp(similarity / self.temperature)
#         numerator = torch.exp(positives / self.temperature)
#         denominator = exp_sim * mask

#         # Compute loss
#         loss = -torch.log(numerator / denominator.sum(dim=1)).mean()

#         return loss




import torch
from torch import nn
from torch.nn import functional as F

class NTXentLoss(nn.Module):
    def __init__(self, temperature=1.0, queue_size=4096):
        """
        [全面升级版] NT-Xent loss -> NNCLR (Nearest Neighbor Contrastive Learning).
        引入了支持队列 (Memory Bank)，将隐空间中最相似的邻居作为正样本，专门为下游密度聚类(HDBSCAN)做强力前置。
        Args:
            temperature (float): scaling factor of the similarity metric.
            queue_size (int): Memory bank 的容量，建议 >= 4096.
        """
        super().__init__()
        self.temperature = temperature
        self.queue_size = queue_size
        
        # 注册队列缓冲，使其随模型存盘，且不参与梯度计算
        self.register_buffer("queue", None)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        # 懒加载初始化 Queue (解决 Init 时无法获知 Embedding 维度的痛点)
        if self.queue is None:
            self.queue = torch.randn(self.queue_size, keys.shape[1], device=keys.device)
            self.queue = F.normalize(self.queue, dim=1)

        # 滚动更新队列
        rem = self.queue_size - ptr
        if rem >= batch_size:
            self.queue[ptr:ptr + batch_size] = keys
            ptr = (ptr + batch_size) % self.queue_size
        else:
            self.queue[ptr:] = keys[:rem]
            self.queue[:batch_size - rem] = keys[rem:]
            ptr = batch_size - rem

        self.queue_ptr[0] = ptr

    def forward(self, z_i, z_j):
        """
        Args:
            z_i (torch.tensor): 原始样本特征 batch
            z_j (torch.tensor): 扰动样本特征 batch (Augmented view)
        """
        batch_size = z_i.size(0)

        # 为了计算 Cosine 相似度并存入队列，先进行 L2 归一化
        z_i_norm = F.normalize(z_i, dim=1)

        # --- [核心改造] 寻找隐空间中的最近邻同学 (Nearest Neighbor) ---
        if self.queue is None:
            # 队列还没建好，第一批数据先存进去，正样本暂时用原本的 z_i
            self._dequeue_and_enqueue(z_i_norm)
            nn_i = z_i 
        else:
            # 与 Memory Bank 计算相似度矩阵 [Batch, Queue_Size]
            sim_with_queue = torch.matmul(z_i_norm, self.queue.T)
            # 找出每个学生在历史数据中最像的邻居的索引
            _, nn_idx = torch.max(sim_with_queue, dim=1)
            # 取出邻居的特征作为正样本 (以此打破原本自监督只能"自己跟自己比"的死局)
            nn_i = self.queue[nn_idx]
            
        # [精妙之处] 将原本公式里的 z_i 偷偷替换成了它的灵魂伴侣 nn_i
        z = torch.cat([nn_i, z_j], dim=0)
        similarity = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

        # 提取对角线上的正样本相似度
        positives = similarity[range(batch_size), range(batch_size, 2 * batch_size)]
        positives = torch.cat([positives, similarity[range(batch_size, 2 * batch_size), range(batch_size)]], dim=0)

        # 创建 Mask 屏蔽自己跟自己的对比
        mask = torch.ones((2 * batch_size, 2 * batch_size), dtype=torch.bool, device=z.device)
        mask.fill_diagonal_(0)

        # 标准 InfoNCE Loss 计算
        exp_sim = torch.exp(similarity / self.temperature)
        numerator = torch.exp(positives / self.temperature)
        denominator = exp_sim * mask

        loss = -torch.log(numerator / denominator.sum(dim=1)).mean()

        # 计算完毕，将当前的特征也推进 Memory Bank 供未来的 Batch 寻找
        self._dequeue_and_enqueue(z_i_norm)

        return loss