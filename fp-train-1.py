import fire  # Expose a simple CLI interface using Python Fire.
import json  # Handle JSON config serialization.
import torch  # Core PyTorch API used by Lightning under the hood.
from data import SyntheticGrandStaffDataset  # Lightning DataModule for synthetic grand staff samples.
from smt_trainer import SMT_Trainer  # LightningModule wrapper around the SMT model.

from ExperimentConfig import experiment_config_from_dict  # Convert raw dict into an ExperimentConfig.
from lightning.pytorch import Trainer  # Orchestrate training and evaluation loops.
from lightning.pytorch.callbacks import ModelCheckpoint  # Save checkpoints driven by monitored metrics.
from lightning.pytorch.loggers import WandbLogger  # Stream metrics to Weights & Biases.
from lightning.pytorch.callbacks.early_stopping import EarlyStopping  # Trigger early exit when metric stalls.

torch.set_float32_matmul_precision('high')  # Improve stability for float16 autocast matmul operations.


def main(config_path):
    """Train and evaluate the SMT fingerprinting model using the provided config."""

    with open(config_path, "r") as f:  # Load the experiment configuration JSON file.
        config = experiment_config_from_dict(json.load(f))  # Materialize a typed ExperimentConfig instance.

    datamodule = SyntheticGrandStaffDataset(config=config.data)  # Build the dataset/datamodule from config.

    max_height = datamodule.get_max_height()  # Determine maximum input image height across splits.
    max_width = datamodule.get_max_width()  # Determine maximum input image width across splits.
    max_len = datamodule.get_max_length()  # Determine maximum target sequence length for decoding.

    model_wrapper = SMT_Trainer(maxh=int(max_height), maxw=int(max_width), maxlen=int(max_len),
                                out_categories=len(datamodule.train_set.w2i),
                                padding_token=datamodule.train_set.w2i["<pad>"],
                                in_channels=1, w2i=datamodule.train_set.w2i, i2w=datamodule.train_set.i2w,
                                d_model=256, dim_ff=256, num_dec_layers=8)  # Instantiate LightningModule tuned to dataset stats.

    group = config.checkpoint.dirpath.split("/")[-1]  # Use checkpoint folder name to group W&B runs.
    wandb_logger = WandbLogger(project='SMT-FP', group=group,
                               name="SMT-System-level", log_model=False)  # Configure Weights & Biases logging.

    early_stopping = EarlyStopping(monitor="val_SER", min_delta=0.01,
                                   patience=5, mode="min", verbose=True)  # Stop if SER does not improve soon enough.

    checkpointer = ModelCheckpoint(dirpath=config.checkpoint.dirpath,
                                   filename=config.checkpoint.filename,
                                   monitor=config.checkpoint.monitor,
                                   mode=config.checkpoint.mode,
                                   save_top_k=config.checkpoint.save_top_k,
                                   verbose=config.checkpoint.verbose)  # Persist best checkpoints per config criteria.

    trainer = Trainer(max_epochs=10000,
                      check_val_every_n_epoch=5,
                      logger=wandb_logger,
                      callbacks=[checkpointer, early_stopping],
                      precision='16-mixed')  # Configure Lightning trainer with callbacks and mixed precision

    trainer.fit(model_wrapper, datamodule=datamodule)  # Start the training loop against the datamodule.

    model = SMT_Trainer.load_from_checkpoint(checkpointer.best_model_path)  # Reload the best checkpoint for testing.

    trainer.test(model, datamodule=datamodule)  # Evaluate the trained model on the test splits.


def launch(config_path):
    main(config_path)  # Provide a Fire-friendly wrapper for the main routine.


if __name__ == "__main__":
    fire.Fire(launch)  # Dispatch CLI execution via Python Fire when run directly.
