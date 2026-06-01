"""Simple EMA (Exponential Moving Average) helper for model parameters.

Stores a shadow copy of parameters (float32) and supports updating and
writing a state dict compatible with `torch.save` for later restoration.
"""

from collections import OrderedDict

import torch


class EMA:
    def __init__(self, model, decay=0.9999, device=None):
        self.decay = float(decay)
        self.device = device
        self.shadow = OrderedDict()
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().cpu().clone().float()

    def update(self, model):
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in self.shadow and p.requires_grad:
                    new = p.detach().cpu().clone().float()
                    self.shadow[name].mul_(self.decay).add_(new, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state):
        self.shadow = OrderedDict((k, v.cpu().clone()) for k, v in state.items())

    def copy_to(self, model):
        """Copy EMA values into the provided model's parameters in-place."""
        for name, p in model.named_parameters():
            if name in self.shadow:
                p.data.copy_(self.shadow[name].to(p.dtype).to(p.device))

    def as_state_dict_for_saving(self, model):
        """Return an OrderedDict matching model.state_dict() but with EMA values for parameters.

        This is convenient to save a checkpoint that can be directly loaded into the model.
        """
        sd = OrderedDict()
        model_sd = model.state_dict()
        for k in model_sd.keys():
            if k in self.shadow:
                sd[k] = self.shadow[k].to(model_sd[k].dtype).clone()
            else:
                sd[k] = model_sd[k].cpu().clone()
        return sd
