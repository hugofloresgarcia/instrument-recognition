import numpy as np
import pytorch_lightning as pl

import torchopenl3

import instrument_recognition as ir
from instrument_recognition import utils

class OpenL3Preprocessor(pl.LightningModule):

    def __init__(self, model_name: str = 'openl3-mel256-6144-music'):
        super().__init__()
        self.save_hyperparameters()
        _, input_repr, embedding_size, content_type = model_name.split('-')
        self.embedding_model = torchopenl3.OpenL3Embedding(input_repr=input_repr, 
                                                    embedding_size=int(embedding_size), 
                                                    content_type=content_type, 
                                                    pretrained=True)
    @classmethod
    def from_hparams(cls, hparams):
        obj = cls.__init__(hparams.preprocessor_name)
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
            audio = utils.effects.augment_from_array_to_array(audio, sr)

        # embed using openl3 model
        embeddings = torchopenl3.embed(model=self.embedding_model, audio=audio, 
                                    sample_rate=sr)
        
        return embeddings