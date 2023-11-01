from typing import TypedDict, Union, Optional

import torch
import torch.nn as nn
from torch.nn import init
from torch_geometric.utils import to_dense_batch
from transformers import AutoModel, BatchEncoding


class ContentsEncoded(TypedDict):
    title: BatchEncoding
    abstract: BatchEncoding
    category: torch.Tensor
    subcategory: torch.Tensor


class MINDBatch(TypedDict):
    batch_hist: torch.Tensor
    batch_cand: torch.Tensor
    x_hist: Union[ContentsEncoded, torch.Tensor]
    x_cand: Union[ContentsEncoded, torch.Tensor]
    targets: Optional[torch.Tensor]


def init_weights(m: nn.Module):
    if isinstance(m, nn.Embedding):
        nn.init.xavier_uniform_(m.weight.data)

    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            init.zeros_(m.bias)

    if isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()


def is_precomputed(x):
    return type(x) is torch.Tensor


class AdditiveAttention(nn.Module):
    def __init__(self, dim=100, r=2.):
        super().__init__()
        intermediate = int(dim * r)
        self.attn = nn.Sequential(
            nn.Linear(dim, intermediate),
            nn.Dropout(0.01),
            nn.LayerNorm(intermediate),
            nn.SiLU(),
            nn.Linear(intermediate, 1),
            nn.Softmax(1),
        )
        self.attn.apply(init_weights)

    def forward(self, context):
        """Additive Attention
        Args:
            context (tensor): [B, seq_len, dim]
        Returns:
            outputs, weights: [B, seq_len, dim], [B, seq_len]
        """
        w = self.attn(context).squeeze(-1)
        return torch.bmm(w.unsqueeze(1), context).squeeze(1), w


class ContentsEncoder(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str,
    ):
        super().__init__()
        bert = AutoModel.from_pretrained(pretrained_model_name)
        self.dim = bert.config.hidden_size

        self.bert = bert
        self.pooler = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.Dropout(0.01),
            nn.LayerNorm(self.dim),
            nn.SiLU(),
        )

        self.pooler.apply(init_weights)

    def forward(self, title: BatchEncoding, **kwargs):
        x_t = self.bert(**title)[0]
        x_t = x_t[:, 0]
        x = self.pooler(x_t)

        return x


class NRMS(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str,
        sa_pretrained_model_name: str,
    ):
        super(NRMS, self).__init__()
        self.encoder = ContentsEncoder(pretrained_model_name)

        dim = self.encoder.dim

        # Use pre-trained self attention.
        # Though this weight is for token level attention,
        # it also works as a good initialization for seq level attention.
        bert = AutoModel.from_pretrained(sa_pretrained_model_name)
        self.self_attn = bert.transformer.layer[-1]  # DistilBERT
        self.additive_attn = AdditiveAttention(dim, 0.5)

    def forward(self,
                batch_hist: torch.Tensor,
                batch_cand: torch.Tensor,
                x_hist: Union[ContentsEncoded, torch.Tensor],
                x_cand: Union[ContentsEncoded, torch.Tensor],
                targets: Optional[torch.Tensor]):
        if not is_precomputed(x_hist):
            x_hist = self.encoder.forward(x_hist)
        x_hist, mask_hist = to_dense_batch(x_hist, batch_hist)
        x_hist = self.self_attn.forward(x_hist, attn_mask=mask_hist)[0]  # DistilBERT
        x_hist, _ = self.additive_attn(x_hist)

        if not is_precomputed(x_cand):
            x_cand = self.encoder.forward(x_cand)
        x_cand, mask_cand = to_dense_batch(x_cand, batch_cand)

        logits = torch.bmm(x_hist.unsqueeze(1), x_cand.permute(0, 2, 1)).squeeze(1)
        logits = logits[mask_cand]

        return logits
        # if targets is None:
        #     return logits
        #
        # if self.training:
        #     criterion = nn.CrossEntropyLoss()
        #     # criterion = LabelSmoothingCrossEntropy()
        #     loss = criterion(logits.reshape(targets.size(0), -1), targets)
        # else:
        #     # In case of val, targets are multi label. It's not comparable with train.
        #     with torch.no_grad():
        #         criterion = nn.BCEWithLogitsLoss()
        #         loss = criterion(logits, targets.float())
        #
        # return loss, logits
