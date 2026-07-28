from typing import Any, List, Tuple

import hydra
import pyrootutils
import torch
from collections import defaultdict
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.nodes import AnyNode

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from src import utils

log = utils.get_pylogger(__name__)


def _configure_checkpoint_loading() -> None:
    """Allowlist OmegaConf classes for PyTorch `weights_only=True` checkpoint loading."""
    torch.serialization.add_safe_globals(
        [
            DictConfig,
            ListConfig,
            ContainerMetadata,
            Metadata,
            AnyNode,
            defaultdict,
            Any,
            bool,
            dict,
            float,
            int,
            list,
            str,
            tuple,
        ]
    )


@utils.task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # Require either ckpt_path (Lightning checkpoint) or an EMA snapshot (EMA pickle).
    # During evaluation, prefer an explicit ema_ckpt_path and fall back to ema_resume_path
    # so existing training-style configs can be reused for inference.
    use_ema = cfg.model.get("use_ema", False)
    ema_ckpt_path = cfg.model.get("ema_ckpt_path")
    ema_resume_path = cfg.model.get("ema_resume_path")
    eval_ema_path = ema_ckpt_path or ema_resume_path
    has_ema = use_ema and (eval_ema_path is not None)
    assert cfg.ckpt_path or has_ema, (
        "Provide either ckpt_path or set model.use_ema=True with "
        "model.ema_ckpt_path/model.ema_resume_path for evaluation."
    )

    if has_ema and ema_ckpt_path is None:
        log.info(
            "Using `model.ema_resume_path` as the EMA source for evaluation because "
            "`model.ema_ckpt_path` is not set."
        )
        cfg.model.ema_ckpt_path = eval_ema_path

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    model._eval_seed = cfg.get("seed")
    la = cfg.get("long_audio")
    model._long_audio_cfg = (
        OmegaConf.to_container(la, resolve=True) if la is not None else {"enabled": False}
    )

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    _configure_checkpoint_loading()
    weights_only = cfg.get("weights_only")

    log.info("Starting testing!")
    # Use ckpt_path only when not loading EMA directly (EMA loads in on_test_start)
    ckpt_path = None if has_ema else cfg.ckpt_path
    trainer.test(
        model=model,
        datamodule=datamodule,
        ckpt_path=ckpt_path,
        weights_only=weights_only,
    )

    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
