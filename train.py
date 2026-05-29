import hydra
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from omegaconf import DictConfig

from src.dataset import MoseiDataModule
from src.utils import set_seed


@hydra.main(version_base=None, config_path=".", config_name="config")
def train(cfg: DictConfig) -> None:
    set_seed(cfg.seed)

    dm = MoseiDataModule(
        **cfg.data,
    )

    model = hydra.utils.instantiate(cfg.model)

    checkpoint_dir = hydra.utils.to_absolute_path(cfg.trainer.checkpoint_dir)
    log_dir = hydra.utils.to_absolute_path(cfg.trainer.log_dir)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best-{epoch:02d}-{val_f1:.3f}",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_f1",
        patience=cfg.trainer.patience,
        mode="max",
    )

    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        accumulate_grad_batches=cfg.trainer.gradient_accumulation_steps,
        logger=[
            CSVLogger(log_dir, name=cfg.logger.experiment_name),
            TensorBoardLogger(log_dir, name=cfg.logger.tensorboard_name),
        ],
        callbacks=[checkpoint_callback, early_stop_callback],
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    train()
