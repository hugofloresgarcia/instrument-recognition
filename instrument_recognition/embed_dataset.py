import os

import numpy as np
import torch

import instrument_recognition.utils as utils

from instrument_recognition.datasets.audio_dataset import AudioDataset
from instrument_recognition.models import torchopenl3

def embed_dataset(path_to_data, path_to_output, 
                 embedding_model_name='openl3-128-512', 
                 batch_size=64, num_workers=18):
    # load our dataset
    dataset = AudioDataset(path_to_data)

    # make a dataloader
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                                         num_workers=num_workers)   

    # get the model
    model = load_embedding_model(embedding_model_name)
    model.freeze().cuda()

    # embed batches
    for batch in loader:
        # get embedding
        X = batch['X'].cuda()
        embedding = model(X)

        # save embedding in numpy format
        for i, metadata_index in enumerate(batch['index']):
            entry = dict(dataset.metadata[metadata_index])

            # get the path to audio and make an embedding path from it 
            path_to_embedding = entry['path_to_audio'].replace(path_to_data, path_to_output)
            assert 'wav' in entry['path_to_audio']
            path_to_embedding = path_to_embedding.replace('.wav', '.npy')
            os.makedirs(path_to_embedding, exist_ok=True)
            entry['path_to_embedding'] = path_to_embedding

            emb_out = embedding[i].detach().cpu().numpy()
            assert emb_out.ndim == 1
            np.save(path_to_embedding, emb_out)

            path_to_entry = path_to_embedding.replace('.npy', '.json')
            utils.data.save_dict_json(entry, path_to_entry)

def load_embedding_model(model_name):
    if 'openl3' in model_name:
        name, n_mels, embedding_size = model_name.split('-')
        model = torchopenl3.OpenL3Embedding(n_mels, embedding_size)
    else:
        raise NameError(f'{model_name} is not a valid model name')

    return model

if __name__=="__main__":
    import argparse
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--path_to_data', type=str, required=True)
    parser.add_argument('--path_to_output', type=str, require=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=20)

    args = parser.parse_args()

    embed_dataset(path_to_data=args.path_to_data, path_to_output=args.path_to_output, 
                  batch_size=args.batch_size, num_workers=args.num_workers)
    