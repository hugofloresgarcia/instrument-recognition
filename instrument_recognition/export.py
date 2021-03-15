"""given a checkpoint path, export the model (as a torchscript file) along with an instruments file"""

from pathlib import Path

import torch
import torchopenl3

import instrument_recognition as ir

class CompositeModel(torch.nn.Module):

    def __init__(self, openl3, head):
        super().__init__()
        self.openl3 = openl3
        self.head = head

    def forward(self, x):
        # assume x is shape (seq, batch, 1, 48000))
        # get embedding tokens
        x = torch.stack([self.openl3(b) for b in x])
        return self.head(x)


def export(exp_path: str, save_path: str):
    # get the best model from checkpoint
    head = ir.utils.train.load_best_model_from_test_tube(exp_path)
    ckpt_path = ir.utils.train.get_best_ckpt_path(Path(exp_path)/'checkpoints')
    ckpt = torch.load(ckpt_path)
    hparams = ckpt['hyper_parameters']

    _, input_repr, embedding_size, content_type = hparams['embedding_name'].split('-')
    
    ## get the torchopenl3 model
    openl3 = torchopenl3.OpenL3Embedding(
        input_repr=input_repr, 
        embedding_size=int(embedding_size), 
        content_type=content_type)

    # concat the head w openl3
    model = CompositeModel(openl3, head)
    model.eval()

    # export to torchscript
    ir.utils.train.save_torchscript_model(model, save_path, torch.randn(10, 1, 1, 48000))
    



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_path', type=str)
    parser.add_argument('--save_path', type=str)

    args = parser.parse_args()

    export(**vars(args))