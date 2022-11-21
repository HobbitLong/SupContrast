import timm
import torch.nn as nn
import torch.nn.functional as F


class SupConVit(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='', head='mlp', feat_dim=128, pretrained=False):
        super(SupConVit, self).__init__()
        dim_in = 768
        print(f"Setting pretrained to {pretrained}")
        self.encoder = timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=0)
        
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat

    def state_dict(self):
        return {k.replace("module.", ""): v for k, v in self.encoder.state_dict().items()}
