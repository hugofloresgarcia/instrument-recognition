import time
import os

import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import torch

def save_torchscript_model(model, save_path):
    """
    saves a pl.LightningModule to torchscript via trace
    """
    audio = torch.randn((10, 1, 48000))

    traced_module = torch.jit.trace(model, audio)
    sm = torch.jit.script(traced_module)
    torch.jit.save(sm, save_path)

def load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:
        weights_dict = np.load(weight_file, allow_pickle=True, encoding='bytes').item()

    return weights_dict

def timing(f):
    """
    wrapper to time a function
    """
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2-time1)*1000.0))

        return ret
    return wrap

def str2bool(value):
    """
    argparse type
    allows you to set bool flags in the following format:
        python program.py --bool1 True --bool2 false ... etc
    """              
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    elif value.lower() in {'none'}:
        return None
    raise ValueError(f'{value} is not a valid boolean value')

def noneint(value):
    if isinstance(value, int):
        return value
    elif isinstance(value, str):
        if value.lower() in {'none',}:
            return None
        else:
            return int(value)
        
    raise ValueError(f'bad input to noneint type')

def get_best_ckpt_path(checkpoint_dir):
    """
    returns the path to the best checkpoint in a directory.
    the checkpoint names need to follow the following format:
        '/{epoch:02d}-{val_loss:.2f}'
    (the precisions don't matter)
    """
    if not os.path.exists(checkpoint_dir): 
        return None
    ckpts = [d for d in os.listdir(checkpoint_dir) if d.split('.')[-1] == 'ckpt']
    if len(ckpts) < 1:
        return None
    ckpts = sorted(ckpts, 
        key=lambda x: float(x.split('-')[-1].split('=')[-1].split('.')[0]))
    best_ckpt = ckpts[0] 
    ckpt_path = os.path.join(checkpoint_dir, best_ckpt)
    print(f'found checkpoint: {ckpt_path}')
    return ckpt_path

def _test_torchscript_model(path_to_model):
    model = torch.jit.load(path_to_model)

    audio = np.randn((3, 1, 48000))

    print(model(audio))

def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]

    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class Hook():
    """
    hook to retrieve the input and output for an nn.Module or pl.LightningModule
    """
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = input[0].detach().cpu()
        self.output = output.detach().cpu()
    def close(self):
        self.hook.remove()

if __name__=="__main__":
    _test_torchscritpt_model()