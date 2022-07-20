
import timm
import torch.nn as nn
import torch.nn.functional as F


class SupConVit(nn.Module):
    """backbone + projection head"""
    def __init__(self, *args, **kwargs):
        super(SupConVit, self).__init__()
        self.encoder = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat

    def state_dict(self):
        return {k.replace("module.", ""): v for k, v in self.encoder.state_dict().items()}
