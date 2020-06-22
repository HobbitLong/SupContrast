"""
Collate function adpated from: 
https://github.com/stanfordmlgroup/chexpert-model/blob/ba51545915a3275b2c9b304b6f701056c092b9ac/data/pad_collate.py
"""

import torch


def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad
    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)


class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)
        return:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
            mask - a mask with 0s in positions that should be ignored
        """
        # find longest sequence
        study_lens = list(map(lambda x: x[0].shape[self.dim], batch))
        max_len = max(study_lens)

        # Pad first example according to max_len
        num_components = max(len(x) for x in batch)
        batch = [(pad_tensor(x[0], pad=max_len, dim=self.dim),) + tuple(x[1:]) for x in batch]

        # Stack padded items and
        batch = tuple(self._merge(batch, component_idx=i) for i in range(num_components))
        masks = [[1] * sl + [0] * (max_len - sl) for sl in study_lens]
        masks = torch.tensor(masks, dtype=torch.float32)

        return batch + (masks,)

    def __call__(self, batch):
        return self.pad_collate(batch)

    @staticmethod
    def _merge(batch, component_idx):
        """Merge components of a batch into a single tensor or list.
        Args:
            batch: Batch to merge.
            component_idx: Index of component in each example that will be merged.
        Returns:
             Merged components
        """
        # Group all components into list
        components = [x[component_idx] for x in batch]
        assert len(components) > 0, 'Error in pad_collate: Cannot merge a batch of size 0'
        first_component = components[0]

        # Merge based on data type of components
        if isinstance(first_component, dict):
            merged_components = {k: [d[k] for d in components] for k in first_component}
        elif isinstance(first_component, torch.Tensor):
            merged_components = torch.stack(components, dim=0)
        else:
            raise ValueError('Unexpected type in PadCollate._merge: {}'.format(type(components[0])))

        return merged_components