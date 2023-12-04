# linearly weighted 2 losses
import torch
from diffusion.diffusion_ddpm_pan import GaussianDiffusion


class LinearlyWeightedLoss(object):
    def __init__(self, warmup_iters, end_weight=1.0) -> None:
        self.warmup_iters = warmup_iters
        self.end_weight = end_weight

        self._linear_weight = lambda x: x / self.warmup_iters * self.end_weight

    def weighted_diffusion_guidance_loss(self, l1, l2, iteration):
        # l1 is diffusion loss, l2 is guidance loss

        if iteration < self.warmup_iters:
            w = self._linear_weight(iteration)
            return l1 * w + l2
        else:
            return l1 + l2


# EMA model
class EmaUpdater(object):
    """exponential moving average model updater
    when iteration > start_iter, update the ema model
    else load the model params to ema model
    """

    def __init__(
        self,
        model: GaussianDiffusion,
        ema_model: GaussianDiffusion,
        decay=0.9999,
        start_iter=0,
    ) -> None:
        self.model = model
        self.ema_model = ema_model
        self.decay = decay
        self.start_iter = start_iter
        self.iteration = start_iter

    @torch.no_grad()
    def update(self, iteration):
        self.iteration = iteration
        if iteration > self.start_iter:
            for p, p_ema in zip(
                self.model.model.parameters(),
                self.ema_model.model.parameters(),
            ):
                p_ema.data = p_ema.data * self.decay + \
                    p.data * (1 - self.decay)
        else:
            for p, p_ema in zip(
                self.model.model.parameters(),
                self.ema_model.model.parameters(),
            ):
                p_ema.data = p.data.clone().detach()

    def load_ema_params(self):
        # load ema params to model
        self.model.load_state_dict(self.ema_model.state_dict())

    def load_model_params(self):
        # load model params to ema model
        self.ema_model.load_state_dict(self.model.state_dict())

    @property
    def on_fly_model_state_dict(self):

        if hasattr(self.model, "module"):
            model = self.model.module.model
        else:
            model = self.model.model

        return model.state_dict()

    @property
    def ema_model_state_dict(self):
        if hasattr(self.model, "module"):
            model = self.ema_model.module.model
        else:
            model = self.ema_model.model

        return model.state_dict()
