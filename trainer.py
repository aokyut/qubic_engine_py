import pytorch_lightning as pl
from model import NNEvaluator, DiscEvaluator
from dataset import QubicDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import os
import argparse


def main(args):
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(os.getcwd(), args.save_dir), filename="nneval-{val_loss:.3f}",
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
        save_last=True
    )
    logger = TensorBoardLogger(save_dir=args.log_dir)

    model = NNEvaluator()
    if args.resume:
        model = model.load_from_checkpoint(args.resume_checkpoint)
        trainer = pl.Trainer(log_every_n_steps=10, 
                                val_check_interval=1000, 
                                callbacks=[checkpoint_callback],
                                resume_from_checkpoint=args.resume_checkpoint,
                                accelerator="auto",
                                devices="auto", 
                                strategy="auto",
                                logger=logger, 
                                )
    else:
        trainer = pl.Trainer(log_every_n_steps=10, 
                                val_check_interval=1000, 
                                callbacks=[checkpoint_callback],
                                accelerator="auto",
                                devices="auto", 
                                strategy="auto",
                                logger=logger, 
                                )

    trainer.fit(model, train_dataloaders=QubicDataModule(
        args.train_data,
        args.valid_data,
        args.valid_data,
        batch_size = 64
    ))

def train_disc(args):
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(os.getcwd(), args.save_dir), filename="nneval-{val_loss/accuracy:.3f}",
        save_top_k=1,
        verbose=True,
        monitor='val_loss/accuracy',
        mode='max',
        save_last=True
    )
    checkpoint_callback2 = ModelCheckpoint(
        dirpath=os.path.join(os.getcwd(), args.save_dir), filename="nneval-{val_loss/entropy:.3f}",
        save_top_k=1,
        verbose=True,
        monitor='val_loss/entropy',
        mode='min',
        save_last=True
    )
    logger = TensorBoardLogger(save_dir=args.log_dir)

    model = DiscEvaluator()
    if args.resume:
        model = model.load_from_checkpoint(args.resume_checkpoint)
        trainer = pl.Trainer(log_every_n_steps=10, 
                                val_check_interval=1000, 
                                callbacks=[checkpoint_callback, checkpoint_callback2],
                                accelerator="auto",
                                devices="auto", 
                                strategy="auto",
                                logger=logger, 
                                resume_from_checkpoint=args.resume_checkpoint
                                )
    else:
        trainer = pl.Trainer(log_every_n_steps=10, 
                                val_check_interval=1000, 
                                callbacks=[checkpoint_callback, checkpoint_callback2],
                                accelerator="auto",
                                devices="auto", 
                                strategy="auto",
                                logger=logger
                                )

    trainer.fit(model, train_dataloaders=QubicDataModule(
        args.train_data,
        args.valid_data,
        args.valid_data, is_disc=True, batch_size=64, label_smoothing=0.2
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data")
    parser.add_argument("valid_data")
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--resume_checkpoint", default=None)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--log_dir", default="lightning_logs")

    args = parser.parse_args()

    main(args)
