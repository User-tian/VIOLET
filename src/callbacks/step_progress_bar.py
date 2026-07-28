"""
Step-based progress bar: shows "Step X/Y" instead of "Epoch N" when training
with max_epochs=-1 (step-based). Use this for curriculum / long step-based runs.
"""
from lightning.pytorch.callbacks import TQDMProgressBar


class StepProgressBar(TQDMProgressBar):
    """Progress bar that displays step progress instead of epoch when max_epochs is -1."""

    def _is_step_based(self, trainer):
        max_epochs = getattr(trainer, "max_epochs", None)
        max_steps = getattr(trainer, "max_steps", None)
        return (max_epochs is not None and max_epochs < 0) and (
            max_steps is not None and max_steps > 0
        )

    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        self._train_bar = bar
        return bar

    def get_metrics(self, trainer, pl_module):
        metrics = super().get_metrics(trainer, pl_module)
        if self._is_step_based(trainer):
            metrics.pop("epoch", None)
        return metrics

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._is_step_based(trainer) and getattr(self, "_train_bar", None) is not None:
            bar = self._train_bar
            max_steps = trainer.max_steps
            gs = trainer.global_step
            bar.total = max_steps
            bar.n = min(gs, max_steps)
            bar.set_description(f"Step {gs}/{max_steps}")
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
