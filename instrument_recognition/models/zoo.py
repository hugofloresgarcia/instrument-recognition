import torch
import torch.nn as nn

from instrument_recognition.models.openl3mlp import OpenL3MLP
from instrument_recognition.models.mlp import MLP512, MLP6144, Ensemble
from instrument_recognition.models.torchopenl3 import OpenL3Embedding
import instrument_recognition.utils as utils


def load_model(model_name, output_units=None, dropout=0.5, random_seed=20):
    """ loads an instrument detection model
    options:  openl3mlp6144-finetuned, openl3mlp-512,
             openl3mlp-6144, mlp-512, mlp-6144
    """
    if model_name == 'baseline-512':
        model = OpenL3MLP(embedding_size=512, dropout=dropout, num_output_units=output_units, 
                          sr=48000, pretrained=False, use_spectrogram_input=True)

    elif model_name == 'baseline-6144':
        model = OpenL3MLP(embedding_size=6144, dropout=dropout, num_output_units=output_units, 
                          sr=48000, pretrained=False, use_spectrogram_input=True)
        
    elif model_name == 'openl3mlp-512':
        model = OpenL3MLP(embedding_size=512, 
                          dropout=dropout,
                          num_output_units=output_units, use_spectrogram_input=True)

    elif model_name == 'openl3mlp-6144':
        model = OpenL3MLP(embedding_size=6144, 
                          dropout=dropout,
                          num_output_units=output_units, use_spectrogram_input=True)

    elif model_name == 'mlp-512':
        model = MLP512(dropout, output_units)

    elif model_name == 'mlp-6144':
        model = MLP6144(dropout, output_units)
        
    elif 'mlp-6144-ensemble-' in model_name:
        num_members = int(model_name.split('-')[-1])
        model_cls = MLP6144
        model = Ensemble(model_cls, num_members, random_seed, 
                         dropout=dropout, num_output_units=output_units)
    
    elif 'mlp-6144-finetuned|' in model_name:
        path_to_state_dict = str(model_name.split('|')[-1])
        embedding = OpenL3Embedding(128, 6144, use_spectrogram_input=True)

        state = torch.load(path_to_state_dict)
        print(state.keys())
        mlp = MLP6144(dropout, output_units)
        mlp.load_state_dict(state)

        model = nn.Sequential(embedding, mlp)

    else:
        raise ValueError(f"couldnt find model name: {model_name}")

    return model
