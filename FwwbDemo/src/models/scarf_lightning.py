"""
SCARF Lightning Module with DAG penalty support.
"""
from typing import Tuple
import torch

from ts3l.pl_modules import TS3LLightining
from ts3l.models import SCARF
from ts3l.utils.scarf_utils import SCARFConfig
from ts3l import functional as F
from ts3l.utils import BaseConfig


class SCARFLightning(TS3LLightining):

    def __init__(self, config: SCARFConfig) -> None:
        """Initialize the pytorch lightining module of SCARF

        Args:
            config (SubTabConfig): The configuration of SCARFLightning.
        """
        super(SCARFLightning, self).__init__(config)

    def _initialize(self, config: BaseConfig):
        """Initializes the model with specific hyperparameters and sets up various components of SCARFLightning.

        Args:
            config (Dict[str, Any]): The given hyperparameter set for SCARF.
        """
        if not isinstance(config, SCARFConfig):
            raise TypeError(f"Expected SCARFConfig, got {type(config)}")

        self.contrastive_loss = NTXentLoss(config.tau)

        self._init_model(SCARF, config)

    def _get_first_phase_loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Calculate the first phase loss

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): The input batch

        Returns:
            torch.Tensor: The final loss of first phase step
        """

        emb_anchor, emb_corrupted = F.scarf.first_phase_step(self.model, batch)

        loss = F.scarf.first_phase_loss(
            emb_anchor, emb_corrupted, self.contrastive_loss
        )

        # ================= [新增核心逻辑] =================
        # 提取当前数据的特征数量(即序列长度 Seq_Len)
        # batch[0] 通常是原始样本 x，shape 为 [Batch_Size, Num_Features]
        seq_len = batch[0].shape[1] 
        
        # 判断当前的 backbone (encoder) 是否是我们魔改后的 Transformer
        if hasattr(self.model.encoder, "get_dag_penalty"):
            # 获取无环约束的惩罚项
            dag_penalty = self.model.encoder.get_dag_penalty(seq_len)
            # 将其作为正则化项加入总 Loss，0.01 是控制因果约束强度的超参数，可以微调
            loss = loss + 0.01 * dag_penalty
        # =================================================

        return loss

    def _get_second_phase_loss(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate the second phase loss

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): The input batch

        Returns:
            torch.Tensor: The final loss of second phase step
            torch.Tensor: The label of the labeled data
            torch.Tensor: The predicted label of the labeled data
        """
        _, y = batch
        y_hat = F.scarf.second_phase_step(self.model, batch)
        task_loss = F.scarf.second_phase_loss(y, y_hat, self.task_loss_fn)

        return task_loss, y, y_hat

    def set_second_phase(self, freeze_encoder: bool = False) -> None:
        """Set the module to fine-tuning

        Args:
            freeze_encoder (bool): If True, the encoder will be frozen during fine-tuning. Otherwise, the encoder will be trainable.
                                    Default is False.
        """
        return super().set_second_phase(freeze_encoder)

    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """The perdict step of SCARF

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): The input batch
            batch_idx (int): For compatibility, do not use

        Returns:
            torch.Tensor: The predicted output (logit)
        """

        y_hat = F.scarf.second_phase_step(self.model, batch)

        return y_hat


# Import NTXentLoss from local losses module
try:
    from .losses import NTXentLoss
except ImportError:
    from losses import NTXentLoss
