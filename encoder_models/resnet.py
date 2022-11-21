
import torch
import timm

from cached_property import cached_property
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from .cluter_base_model import ClusterBaseModel as BaseModel


class Resnet(BaseModel):

    NAME = 'resnet'

    @cached_property
    def transform(self):
        config = resolve_data_config({}, model=self.model)
        transform = create_transform(**config)
        return transform

    def __init__(self, weights_path=""):
        """Initialize"""
        super().__init__(weights_path)
        model = timm.create_model('resnet50', pretrained=True, num_classes=0)
        model.eval()
        self.model = model

    def encode(self, preprocessed_image):
        r = self.model(torch.unsqueeze(preprocessed_image, 0))
        return r.detach().numpy()
