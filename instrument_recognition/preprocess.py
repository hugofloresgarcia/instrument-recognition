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

def load_model_from_str(name: str):
    name = name.split('-')[0]

    if name == 'openl3':
        _, input_repr, embedding_size, content_type = name.split('-')
        model = torchopenl3.OpenL3Embedding(input_repr=input_repr, embedding_size=int(embedding_size), 
                                        content_type=content_type, pretrained=True)
    if name == 'vggish':
        model = torch.hub.load('harritaylor/torchvggish', 'vggish')
        model.eval()
    else:
        raise ValueError
    
    return model

def _preprocess_openl3(model, name: str, model_name: str,  batch_size: int, num_workers: int, device: int = None):
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
            #NOTE: BAD IDEA to hardcode this 10
            X = X.view(batch_size, 10, 1, X.shape[-1])
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

def _preprocess_vggish(model, name:str, model_name: str, num_workers: int, device:int = None):
    model.to(device)
    dm = ir.datasets.DataModule(name, 1, num_workers)
    dm.setup()

    for dataset in (dm.train_data, dm.test_data):
        pbar = tqdm.tqdm(range(len(dataset)))
        for idx in pbar:
            record = dataset[idx]
            path = record['path_to_audio']
            with torch.no_grad():
                embedding = model.forward(path).detach().cpu().numpy()
            
            pbar.set_description(str(Path(path).stem))
            cache_dir = ir.CACHE_DIR / model_name
            path_to_cached_file = cache_dir / Path(path).relative_to(ir.DATA_DIR).with_suffix('.npy')

            # save embeddings to cache
            os.makedirs(path_to_cached_file.parent, exist_ok=True)
            np.save(str(path_to_cached_file), embedding)

def preprocess_dataset(name: str, model_name: str, batch_size: int, num_workers: int, device: int = None):
    model = load_model_from_str(model_name)
    if device is not None:
        model = model.to(device)
    if 'openl3' in model_name:
        _preprocess_openl3(model, name=name, model_name=model_name, batch_size=batch_size, num_workers=num_workers, device=device)
    elif 'vggish' in model_name:
        _preprocess_vggish(model, name=name, model_name=model_name, num_workers=num_workers, device=device)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, nargs='+')
    parser.add_argument('--model', type=str, nargs='+')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=18)
    parser.add_argument('--device', type=int, default=0)

    args = parser.parse_args()
    models = ALL_MODEL_NAMES if args.model == 'all' else args.model
    for model in models:
        for name in args.name:
            preprocess_dataset(name, model, args.batch_size, args.num_workers, args.device)