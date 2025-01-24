from argparse import ArgumentParser
import pdb
import yaml

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from scipy.stats import spearmanr, pearsonr
import numpy as np
from tqdm import tqdm
import logging
from trainer import AutoEncoder, retrieve_encoder, retrieve_decoder

def main(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model = AutoEncoder.load_from_checkpoint(config['checkpoint'])
    print('Model loaded')
    for task in config['eval_tasks']:
        if task == "stsb":
            print('Evaluating on STS-B')
            from data_loaders import STSBDataset, STSBEvaluator
            evaluator = STSBEvaluator(model)

        elif task == "paws":
            print('Evaluating on PAWS')
            from data_loaders import PAWSDataset, PAWSFineTuner
            evaluator = PAWSFineTuner(model)

            # Create trainer
            finetuner = L.Trainer(
                max_epochs=3,
                accelerator='gpu',
                devices=config['devices'],
                strategy=config['mgpu-strategy'],
                callbacks=[
                    ModelCheckpoint(
                        monitor='val_f1',
                        mode='max',
                        save_top_k=1,
                        filename='best-paws-model'
                    ),
                    EarlyStopping(
                        monitor='train_loss',
                        mode='max',
                        patience=2
                )
                ],
                logger = None
            )

            # Fine-tune
            finetuner.fit(evaluator)

        logger = CSVLogger(save_dir=config['log_dir'], name=task)
        trainer = L.Trainer(logger = logger,
            devices = config['devices'],
            strategy = config['mgpu-strategy'],
            accelerator="gpu")

        # Run evaluation
        trainer.test(evaluator)

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)