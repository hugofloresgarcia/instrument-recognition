from pathlib import Path
import os

import numpy as np
import tqdm
import pytorch_lightning as pl
import torch

import torchopenl3
import audio_utils as au

import instrument_recognition as ir
import instrument_recognition.datasets 
from instrument_recognition import utils

ALL_MODEL_NAMES = ('openl3-mel128-6144-music', 'openl3-mel256-6144-music',
                  'openl3-mel128-512-music', 'openl3-mel256-512-music',
                  'openl3-mel128-6144-env', 'openl3-mel256-6144-env',
                  'openl3-mel128-512-env', 'openl3-mel256-512-env')

class OpenL3Preprocessor(pl.LightningModule):

    def __init__(self, model_name: str = 'openl3-mel256-6144-music', cuda_device=None):
        super().__init__()
        self.save_hyperparameters()
        self.name = model_name
        _, input_repr, embedding_size, content_type = model_name.split('-')
        self.embedding_model = torchopenl3.OpenL3Embedding(input_repr=input_repr, 
                                                    embedding_size=int(embedding_size), 
                                                    content_type=content_type, 
                                                    pretrained=True)
        
        self.cuda_device = cuda_device
    
    @classmethod
    def from_hparams(cls, hparams):
        obj = cls(hparams.preprocessor_name)
        obj.hparams = hparams
        return obj
    
    @classmethod
    def add_argparse_args(cls, parent_parser):
        parser = parent_parser
        parser.add_argument('--preprocessor_name', type=str, default='openl3-mel256-6144-music')
        return parser
    
    def __call__(self, audio, sr, augment=False):
        # add augmentation here?
        if augment:
            audio, effect_params = utils.effects.augment_from_array_to_array(audio, sr)

        # embed using openl3 model
        embeddings = torchopenl3.embed(model=self.embedding_model, audio=audio, 
                                    sample_rate=sr, device=self.cuda_device)
        
        return embeddings

def preprocess_dataset(name: str, model_name: str, batch_size: int, num_workers: int, device: int = None):
    _, input_repr, embedding_size, content_type = model_name.split('-')
    model = torchopenl3.OpenL3Embedding(input_repr=input_repr, embedding_size=int(embedding_size), 
                                        content_type=content_type, pretrained=True)
    if device is not None:
        model = model.to(device)

    dm = ir.datasets.DataModule(name, batch_size, num_workers)
    dm.setup()

    loaders = ((dm.train_dataloader(), True, dm.train_data.root_dir), 
                (dm.val_dataloader(), True,  dm.val_data.root_dir))

    for loader, augment,  root_dir in loaders:
        pbar = tqdm.tqdm(loader)
        for batch in pbar:
            # get all inputs
            X = batch['X']

            # get embedding
            X = X.view(-1, 10, 1, 48000)
            X = X.permute(1, 0, 2, 3)
            X_embedded = []
            for X_step in X:
                if device is not None:
                    X_step = X_step.to(device)
                X_step = model(X_step).detach().cpu().numpy()
                X_embedded.append(X_step)
            X = np.stack(X_embedded)
            X = X.transpose(1, 0, 2)

            # now, iterate through paths
            for x, path in zip(X, batch['path_to_audio']):
                pbar.set_description(str(Path(path).stem))
                cache_dir = ir.CACHE_DIR / model_name
                path_to_cached_file = cache_dir / Path(path).relative_to(ir.DATA_DIR).with_suffix('.npy')

                # save embeddings to cache
                os.makedirs(path_to_cached_file.parent, exist_ok=True)
                np.save(str(path_to_cached_file), x)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=18)
    parser.add_argument('--device', type=int, default=0)

    args = parser.parse_args()
    if args.model == 'all':
        for model in ALL_MODEL_NAMES:
            preprocess_dataset(args.name, model, args.batch_size, args.num_workers, args.device)
    else:
        preprocess_dataset(args.name, args.model, args.batch_size, args.num_workers, args.device)

    
    