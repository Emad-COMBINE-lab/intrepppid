import pytorch_lightning as pl
from torch import nn
import torchmetrics
from intrepppid.utils import embedding_dropout, DictLogger
import torch
from ranger21 import Ranger21
from typing import Dict, Any, Optional
from pathlib import Path
from intrepppid.encoders import make_barlow_encoder
from intrepppid.classifier.head import MLPHead
from intrepppid.data import RapppidDataModule2
import json
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything
from os import makedirs


class ClassifyNet(pl.LightningModule):
    def __init__(
        self,
        encoder,
        head,
        embedding_droprate: float,
        num_epochs: int,
        steps_per_epoch: int,
        fine_tune_epochs: Optional[int] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.embedding_droprate = embedding_droprate
        self.criterion = nn.BCEWithLogitsLoss()
        self.num_epochs = num_epochs
        self.steps_per_epoch = steps_per_epoch
        self.fine_tune_epochs = fine_tune_epochs

        self.auroc = torchmetrics.AUROC(task="binary")
        self.average_precision = torchmetrics.AveragePrecision(task="binary")

        self.do_rate = 0.3
        self.head = head

    def embedding_dropout(self, embed, words, p=0.2):
        return embedding_dropout(self.training, embed, words, p)

    def forward(self, x1, x2):
        if (
            self.fine_tune_epochs is not None
            and self.current_epoch >= self.num_epochs - self.fine_tune_epochs
        ):
            x1 = self.encoder(x1)
            x2 = self.encoder(x2)
        else:
            with torch.no_grad():
                x1 = self.encoder(x1)
                x2 = self.encoder(x2)

        y_hat = self.head(x1, x2)

        return y_hat

    def step(self, batch, stage):
        x1, x2, y = batch

        y_hat = self(x1, x2).squeeze(1)

        loss = self.criterion(y_hat, y.float())

        self.log(f"{stage}_loss", loss, on_epoch=True, on_step=False)

        auroc = self.auroc(y_hat, y)
        self.log(f"{stage}_auroc", auroc, on_epoch=True, on_step=False)

        ap = self.average_precision(y_hat, y)
        self.log(f"{stage}_ap", ap, on_epoch=True, on_step=False)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")

    def configure_optimizers(self):
        optimizer = Ranger21(
            self.parameters(),
            lr=1e-3,
            weight_decay=1e-4,
            num_batches_per_epoch=self.steps_per_epoch,
            num_epochs=self.num_epochs,
            warmdown_start_pct=0.72,
        )
        return optimizer


def get_classifynet(hyperparams: Optional[Dict[str, Any]]):
    architecture = hyperparams["architecture"]
    del hyperparams[architecture]


def train_classifier_barlow(
    barlow_hyperparams_path: Path,
    barlow_checkpoint_path: Path,
    ppi_dataset_path: Path,
    log_path: Path,
    hyperparams_path: Path,
    chkpt_dir: Path,
    c_type: int,
    model_name: str,
    workers: int,
    embedding_droprate: float,
    do_rate: float,
    num_epochs: int,
    batch_size: int,
    seed: Optional[int] = None,
    fine_tune_epochs: Optional[int] = None,
):
    makedirs(chkpt_dir, exist_ok=True)
    makedirs(log_path, exist_ok=True)
    makedirs(hyperparams_path.parent, exist_ok=True)

    with open(barlow_hyperparams_path) as f:
        barlow_hyperparams = json.load(f)

    seed = barlow_hyperparams["seed"] if seed is None else seed

    seed_everything(seed)

    hyperparameters = {
        "architecture": "ClassifierBarlow",
        "barlow_hyperparams_path": str(barlow_hyperparams_path),
        "barlow_checkpoint_path": str(barlow_checkpoint_path),
        "ppi_dataset_path": str(ppi_dataset_path),
        "log_path": str(log_path),
        "hyperparams_path": str(hyperparams_path),
        "chkpt_dir": str(chkpt_dir),
        "model_name": model_name,
        "workers": workers,
        "embedding_droprate": embedding_droprate,
        "do_rate": do_rate,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "seed": seed,
        "fine_tune_epochs": fine_tune_epochs,
    }

    with open(hyperparams_path, "w") as f:
        json.dump(hyperparameters, f)

    data_module = RapppidDataModule2(
        batch_size=batch_size,
        dataset_path=ppi_dataset_path,
        c_type=c_type,
        trunc_len=barlow_hyperparams["trunc_len"],
        workers=workers,
        vocab_size=barlow_hyperparams["vocab_size"],
        model_file=barlow_hyperparams["model_path"],
        seed=seed,
    )

    data_module.setup("training")
    steps_per_epoch = len(data_module.train_dataloader())

    encoder = make_barlow_encoder(
        barlow_hyperparams["vocab_size"],
        barlow_hyperparams["embedding_size"],
        barlow_hyperparams["rnn_num_layers"],
        barlow_hyperparams["rnn_dropout_rate"],
        barlow_hyperparams["variational_dropout"],
        barlow_hyperparams["bi_reduce"],
        barlow_hyperparams["batch_size"],
        barlow_hyperparams["embedding_droprate"],
        barlow_hyperparams["num_epochs"],
        steps_per_epoch,
    )

    weights = torch.load(barlow_checkpoint_path)["state_dict"]

    encoder.load_state_dict(weights)
    encoder.eval()

    head = MLPHead(barlow_hyperparams["embedding_size"], do_rate)

    net = ClassifyNet(
        encoder, head, embedding_droprate, num_epochs, steps_per_epoch, fine_tune_epochs
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=chkpt_dir,
        filename=model_name + "-{epoch:02d}-{val_loss:.2f}",
    )

    dict_logger = DictLogger()

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=num_epochs,
        precision=16,
        logger=[dict_logger],
        callbacks=[checkpoint_callback],
    )

    trainer.fit(net, data_module)

    test_results = trainer.test(dataloaders=data_module, ckpt_path="best")

    dict_logger.metrics["test_results"] = test_results

    with open(log_path / model_name / "metrics.json", "w") as f:
        json.dump(dict_logger.metrics, f, indent=3)
