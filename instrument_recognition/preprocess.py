import os
from concurrent.futures import ThreadPoolExecutor

import torch
import tqdm
import numpy as np

from instrument_recognition.datasets import load_datamodule, debatch
from instrument_recognition import utils

PATH_TO_DATA = '/home/hugo/data/mono_music_sed/mdb/AUDIO/'
CUDA_DEVICE = 0

# define our preprocess callables and assign them a path
preprocessors = [
    {'path': '/home/hugo/data/mono_music_sed/mdb/EMBEDDINGS/', 
    'model': OpenL3Embedding(n_mels=128, embedding_size=6144, pretrained=True).cuda(CUDA_DEVICE)}, 
    # {'path': '/home/hugo/data/mono_music_sed/mdb/SPECTROGRAMS/', 
    # 'model': Melspectrogram(sr=48000, n_mels=128).cuda(CUDA_DEVICE)}
]

if __name__ == "__main__":
    for preprocessor in preprocessors:
        print(f'saving to: {preprocessor["path"]}')
        # load a base datamodule and 
        # let it do the magic  
        dm = load_datamodule(PATH_TO_DATA, batch_size=64, num_workers=2, use_npy=False)

        # make dataloader - dataset pairs
        pairs = ((dm.train_dataloader(), dm.train_data, 'train'), 
                (dm.test_dataloader(), dm.test_data, 'test'),
                (dm.val_dataloader(), dm.val_data, 'validation'))

        model = preprocessor['model']
        model.eval()

        # iterate through the dataloaders
        for dl, dataset, subset_name in pairs:
            pbar = tqdm.tqdm(dl)
            for batch in pbar:
                # retrieve what we need from the dataloader
                X = batch['X'].cuda(CUDA_DEVICE)

                # forward pass through the model
                with torch.no_grad():
                    preprocessed_X = model(X).detach().cpu().numpy()

                # subpbar = tqdm.tqdm(enumerate(zip(preprocessed_X, batch['metadata_index'])))
                
                def save_data_and_metadata(index_tuples):
                    batch_idx, metadata_index = index_tuples
                    x = preprocessed_X[batch_idx]
                    # create a new metadata dict for our preprocessed data
                    new_entry = dict(dataset.metadata[metadata_index])

                    base_chunk_name = new_entry['base_chunk_name']
                    start_time = new_entry['start_time']
                    label = new_entry['label']

                    # double check we doing the right thing
                    assert new_entry['path_to_audio'] == batch['path_to_audio'][batch_idx]

                    # create new paths!
                    new_entry['path_to_npy'] = os.path.join(preprocessor['path'], subset_name, label,
                                                        base_chunk_name, f'{start_time}.npy')
                    new_entry['path_to_metadata'] = new_entry['path_to_npy'].replace('.npy', '.yaml')
                    os.makedirs(os.path.dirname(new_entry['path_to_npy']), exist_ok=True)
                    # print(f'saving {new_entry["path_to_npy"]}')
                    
                    # save the new things!
                    utils.data.save_dict_yaml(new_entry, new_entry['path_to_metadata'])
                    np.save(new_entry['path_to_npy'], x)
                
                # multithreading this will make it SO MUCH FASTER (IO bound problem)
                index_tuples = [(b, m) for b, m in enumerate(batch['metadata_index'])]
                with ThreadPoolExecutor(max_workers=40) as executor:
                    fut = executor.map(save_data_and_metadata, index_tuples)
                    _ = list(fut)
                

