"""
Special LR scheduler for fine-tuning Transformer networks
"""

import logging

import torch
from allennlp.training.learning_rate_schedulers.learning_rate_scheduler import (
    LearningRateScheduler,
)

logger = logging.getLogger(__name__)


@LearningRateScheduler.register("ulmfit_sqrt")
class UlmfitSqrtLR(LearningRateScheduler):
    """Implements a combination of ULMFiT (slanted triangular) with Noam sqrt learning rate decay"""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model_size: int,
        warmup_steps: int,
        start_step: int = 0,
        factor: float = 100,
        steepness: float = 0.5,
        last_epoch: int = -1,
        gradual_unfreezing: bool = False,
        discriminative_fine_tuning: bool = False,
        decay_factor: float = 0.38,
    ) -> None:
        self.warmup_steps = warmup_steps + start_step
        self.start_step = start_step
        self.factor = factor
        self.steepness = steepness
        self.model_size = model_size
        self.gradual_unfreezing = gradual_unfreezing
        self.freezing_current = self.gradual_unfreezing

        if self.gradual_unfreezing:
            assert not optimizer.param_groups[-1][
                "params"
            ], "The default group should be empty."
        if self.gradual_unfreezing or discriminative_fine_tuning:
            assert len(optimizer.param_groups) > 2, (
                "There should be at least 3 param_groups (2 + empty default group)"
                " for gradual unfreezing / discriminative fine-tuning to make sense."
            )

        super().__init__(optimizer, last_epoch=last_epoch)

        if discriminative_fine_tuning:
            # skip the last param_group if it is has no parameters
            exponent = 0
            for i in range(len(self.base_values) - 1, -1, -1):
                param_group = optimizer.param_groups[i]
                if param_group["params"]:
                    param_group["lr"] = self.base_values[i] * decay_factor ** exponent
                    self.base_values[i] = param_group["lr"]
                    exponent += 1

    def step(self, metric: float = None, epoch: int = None) -> None:
        if self.gradual_unfreezing:
            # the method is called once when initialising before the
            # first epoch (epoch -1) and then always at the end of each
            # epoch; so the first time, with epoch id -1, we want to set
            # up for epoch #1; the second time, with epoch id 0,
            # we want to set up for epoch #2, etc.
            num_layers_to_unfreeze = epoch + 2 if epoch > -1 else 1
            if num_layers_to_unfreeze >= len(self.optimizer.param_groups) - 1:
                logger.info("Gradual unfreezing finished. Training all layers.")
                self.freezing_current = False
            else:
                logger.info(
                    f"Gradual unfreezing. Training only the top {num_layers_to_unfreeze} layers."
                )
            for i, param_group in enumerate(reversed(self.optimizer.param_groups)):
                for param in param_group["params"]:
                    # i = 0 is the default group; we care about i > 0
                    param.requires_grad = bool(i <= num_layers_to_unfreeze)

    def step_batch(self, batch_num_total: int = None) -> None:
        if batch_num_total is None:
            self.last_epoch += 1  # type: ignore
        else:
            self.last_epoch = batch_num_total
        for param_group, learning_rate in zip(
            self.optimizer.param_groups, self.get_values()
        ):
            param_group["lr"] = learning_rate

    def get_values(self):
        if self.freezing_current:
            # If parameters are still frozen, keep the base learning rates
            return self.base_values

        # This computes the Noam Sqrt LR decay based on the current step
        step = max(self.last_epoch - self.start_step, 1)
        scale = self.factor * (
            self.model_size ** (-0.5)
            * min(
                step ** (-self.steepness),
                step * self.warmup_steps ** (-self.steepness - 1),
            )
        )

        return [scale * lr for lr in self.base_values]
