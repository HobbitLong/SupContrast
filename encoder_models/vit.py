import typing

import torch
import timm

from cached_property import cached_property
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from .cluter_base_model import ClusterBaseModel
from .rank_base_model import RankBaseModel


class Vit:

    @cached_property
    def transform(self):
        config = resolve_data_config({}, model=self.model)
        config["mean"] = self.mean
        config["std"] = self.std
        transform = create_transform(**config)
        return transform

    def __init__(self, weights_path, mean: typing.List[float], std: typing.List[float], question_dir: str):
        """Initialize"""
        super().__init__(weights_path)
        self.std = std
        self.mean = mean
        model = timm.create_model('vit_base_patch16_224')
        weights = torch.load(self.weights_path, map_location=torch.device('cuda'))
        model.load_state_dict({k.replace("module.", ""): v for k, v in weights["model"].items()}, strict=False)
        model.to("cuda")
        model.eval()
        self.model = model
        self.question_dir = question_dir

    def encode(self, preprocessed_images):
        r = self.model.forward_features(preprocessed_images.to("cuda"))
        return r.cpu().detach().numpy()


class ClusterVit(Vit, ClusterBaseModel):

    NAME = 'cluster_vit'


class RankVit(Vit, RankBaseModel):

    NAME = 'rank_vit'
