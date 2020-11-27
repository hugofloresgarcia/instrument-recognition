import os

import numpy as np
import torch
import tqdm
import openl3

import instrument_recognition.utils as utils

from instrument_recognition.datasets.base_dataset import BaseDataset
from instrument_recognition.models import torchopenl3


def get_original_openl3_embedding(model, X):
    e =[]
    for x in X:
        emb, _ = openl3.get_audio_embedding(x[0].detach().cpu().numpy(), 48000, model, verbose=False)
        emb = torch.from_numpy(emb)
        e.append(emb[0])
    e = torch.stack(e)
    assert e.ndim == 2
    return e

def embed_dataset(path_to_data, path_to_output, 
                 embedding_model_name='openl3-128-512', 
                 batch_size=64, num_workers=18):
    # load our dataset
    dataset = BaseDataset(path_to_data, use_embeddings=False)

    # make a dataloader
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                                         num_workers=num_workers)

    # get the model
    model = load_embedding_model(embedding_model_name)
    model.eval()
    model.cuda()

    # import openl3 
    # openl3_model = openl3.models.load_audio_embedding_model("mel128", content_type="music", embedding_size=512)

    # embed batches
    pbar = tqdm.tqdm(loader)
    for batch in pbar:
        # get embedding
        X = batch['X'].cuda()
        with torch.no_grad():
            embedding = model(X)
        # embedding = get_original_openl3_embedding(openl3_model, X)

        # save embedding in numpy format
        for i, metadata_index in enumerate(batch['metadata_index']):
            entry = dict(dataset.metadata[metadata_index])

            # get the path to audio and make an embedding path from it 
            path_to_embedding = entry['path_to_audio'].replace(path_to_data, path_to_output+'/')
            assert 'wav' in entry['path_to_audio']
            path_to_embedding = path_to_embedding.replace('.wav', '.npy')
            os.makedirs(os.path.dirname(path_to_embedding), exist_ok=True)
            entry['path_to_embedding'] = path_to_embedding

            emb_out = embedding[i].detach().cpu().numpy()
            assert emb_out.ndim == 1
            np.save(path_to_embedding, emb_out)

            path_to_entry = path_to_embedding.replace('.npy', '.yaml')
            utils.data.save_dict_yaml(entry, path_to_entry)

def load_embedding_model(model_name):
    if 'openl3' in model_name:
        name, n_mels, embedding_size = model_name.split('-')
        model = torchopenl3.OpenL3Embedding(int(n_mels), int(embedding_size))
    else:
        raise NameError(f'{model_name} is not a valid model name')

    return model

if __name__=="__main__":
    import argparse
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--path_to_data', type=str, required=True)
    parser.add_argument('--path_to_output', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=20)
    parser.add_argument('--model_name', type=str, default='openl3-128-512')

    args = parser.parse_args()

    embed_dataset(path_to_data=args.path_to_data, path_to_output=args.path_to_output, 
                  batch_size=args.batch_size, num_workers=args.num_workers, 
                  embedding_model_name=args.model_name)
    