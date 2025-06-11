import torch
import torch.nn as nn
from transformers import AutoModel

class SupConXLMRLarge(nn.Module):
    """XLM-RoBERTa-large backbone + projection head"""
    def __init__(self, head='mlp', feat_dim=128):
        super(SupConXLMRLarge, self).__init__()
        self.encoder = AutoModel.from_pretrained("xlm-roberta-large")
        dim_in = 1024  # XLM-RoBERTa-large embedding size
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError('head not supported: {}'.format(head))

    def forward(self, input_ids, attention_mask=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        feat = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
        feat = nn.functional.normalize(self.head(feat), dim=1)  # L2 normalize
        return feat