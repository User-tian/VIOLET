import torch
from torch.utils.data import IterableDataset, Dataset, ConcatDataset
import random
import numpy as np
from typing import Dict, List, Optional

# Keys matching violin_datamodule's real_keys / synth_keys
_REAL_KEYS = ["musc", "mosa_real"]
_SYNTH_KEYS = ["violet", "mosa_vpt"]


class _EmptyDataset(Dataset):
    """Placeholder for when no real/synth datasets exist."""

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int):
        raise IndexError("Empty dataset has no items")


class UnifiedViolinDataset(IterableDataset):
    def __init__(self,
                 datasets: Dict[str, Dataset],
                 batch_size: int,
                 stage1_steps: int,
                 stage_transition_steps: int,
                 stage1_ratios: Dict[str, float],
                 stage2_ratios: Dict[str, float],
                 accumulate_grad_batches: int = 1,
                 seed: int = 42
                 ):
        """
        Args:
            datasets: Dictionary of name -> Dataset.
            stage1_ratios: Dictionary of name -> probability (sum to 1.0).
            stage2_ratios: Dictionary of name -> probability (sum to 1.0).
        """
        self.datasets = datasets
        self.batch_size = batch_size
        self.stage1_steps = stage1_steps
        self.stage_transition_steps = max(int(stage_transition_steps), 0)
        self.stage1_ratios = stage1_ratios
        self.stage2_ratios = stage2_ratios
        self.accumulate_grad_batches = max(int(accumulate_grad_batches), 1)
        self.seed = seed

        # Only include non-empty datasets in the sampling pool (avoid randint(0,0) error)
        all_keys = sorted(list(datasets.keys()))
        self.dataset_keys = [k for k in all_keys if len(datasets[k]) > 0]
        self.dataset_lens = {k: len(d) for k, d in datasets.items()}

        empty_keys = [k for k in all_keys if k not in self.dataset_keys]
        if empty_keys:
            import warnings
            warnings.warn(
                f"UnifiedViolinDataset: Excluding empty dataset(s) from sampling: {empty_keys}. "
                f"Check that data paths exist and contain valid samples."
            )
        if not self.dataset_keys:
            raise ValueError(
                "UnifiedViolinDataset: All datasets are empty. "
                "Check data_dir, mosa_dir, musc_dir, mosa_real_dir paths and that they contain train splits."
            )

        # Pre-compute weights arrays for efficient sampling (only for non-empty datasets)
        self.s1_weights = np.array([stage1_ratios.get(k, 0.0) for k in self.dataset_keys])
        self.s2_weights = np.array([stage2_ratios.get(k, 0.0) for k in self.dataset_keys])

        # Normalize (weights for empty datasets are already 0, so sum may be < 1)
        s1_sum = self.s1_weights.sum()
        s2_sum = self.s2_weights.sum()
        self.s1_weights /= (s1_sum + 1e-8)
        self.s2_weights /= (s2_sum + 1e-8)

        # Expose real/synth datasets for anchor-audio sampling (violin_diffusion_module.on_train_start)
        real_list = [datasets[k] for k in _REAL_KEYS if k in datasets and len(datasets[k]) > 0]
        synth_list = [datasets[k] for k in _SYNTH_KEYS if k in datasets and len(datasets[k]) > 0]
        self.real_dataset = ConcatDataset(real_list) if real_list else _EmptyDataset()
        self.synth_dataset = ConcatDataset(synth_list) if synth_list else _EmptyDataset()

    def configure_schedule(self, accumulate_grad_batches: int = 1) -> None:
        self.accumulate_grad_batches = max(int(accumulate_grad_batches), 1)

    def _estimated_optimizer_step(self, local_step: int, num_workers: int) -> float:
        samples_per_optimizer_step = max(
            1,
            self.batch_size * self.accumulate_grad_batches,
        )
        return (local_step * num_workers) / float(samples_per_optimizer_step)

    def _weights_for_step(self, optimizer_step_est: float) -> np.ndarray:
        if self.stage_transition_steps <= 0:
            return self.s1_weights if optimizer_step_est < self.stage1_steps else self.s2_weights

        if optimizer_step_est < self.stage1_steps:
            return self.s1_weights

        transition_end = self.stage1_steps + self.stage_transition_steps
        if optimizer_step_est >= transition_end:
            return self.s2_weights

        alpha = (optimizer_step_est - self.stage1_steps) / float(self.stage_transition_steps)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        return (1.0 - alpha) * self.s1_weights + alpha * self.s2_weights

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0

        # Seed for reproducibility
        # Use different seed per worker
        rng = random.Random(self.seed + worker_id)
        np_rng = np.random.RandomState(self.seed + worker_id)

        local_step = 0

        while True:
            # Estimate optimizer step from the number of yielded samples.
            optimizer_step_est = self._estimated_optimizer_step(
                local_step=local_step,
                num_workers=num_workers,
            )
            weights = self._weights_for_step(optimizer_step_est)

            # Select dataset
            # Use numpy choice which is fast for weighted selection
            dataset_idx = np_rng.choice(len(self.dataset_keys), p=weights)
            dataset_key = self.dataset_keys[dataset_idx]

            # Select sample from dataset
            # We assume datasets are map-style (indexable)
            ds = self.datasets[dataset_key]
            idx = np_rng.randint(0, len(ds))

            yield ds[idx]

            local_step += 1
