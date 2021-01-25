import time
import os

import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import torch

import instrument_recognition as ir

def load_best_model_from_test_tube(test_tube_dir: str):
    """because the model is wrapped in a Task object, 
    getting the best model is not trivial. Given a test tube dir
    (along with name and version), this function will load a Model
    with its state dict. 
    """
    ckpt = ir.utils.train.get_best_ckpt_path(test_tube_dir / 'checkpoints')
    print(test_tube_dir)
    print(ckpt)

    ckpt = torch.load(ckpt)

    def strip_state_dict_keys(state_dict, pattern='model.'):
        from collections import OrderedDict
        # remove the pattern from the state dict keys
        output = OrderedDict()
        for k in state_dict:
            if pattern in k:
                new_k = k.replace(pattern, '')
                output[new_k] = ckpt['state_dict'][k]
        return output

    model = ir.models.Model(**ckpt['hyper_parameters'])
    model.load_state_dict(strip_state_dict_keys(ckpt['state_dict'], 'model.'))
    return model

def save_torchscript_model(model, save_path, example_input):
    """
    saves a module to torchscript via trace
    """
    # move to cpu if we need to
    model = model.cpu()
    example_input = example_input.cpu()
    traced_module = torch.jit.trace(model, example_input)
    
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

def mixup_data(x, y, alpha=1.0, dim=0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[dim]
    index = torch.randperm(batch_size).type_as(x).long()

    mixed_x = lam * x + (1 - lam) * torch.index_select(x, dim=dim, index=index).type_as(x)
    y_a, y_b = y, torch.index_select(y, dim=dim, index=index).type_as(y)
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