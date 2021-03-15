from multiprocessing import Value
import os

import yaml
import torch
import torch.nn as nn
import pytorch_lightning as pl

import instrument_recognition as ir
import instrument_recognition.utils as utils
from instrument_recognition.task import InstrumentDetectionTask, train_instrument_detection_model
from instrument_recognition.models import Model
from instrument_recognition.datasets import DataModule

def dump_classlist(dm, save_dir):
    # get classlist and number of classes
    classlist = dm.classlist
    print(f'classlist is: {classlist}')
    with open(os.path.join(save_dir, 'classlist.yaml'), 'w') as f:
        yaml.dump(classlist, f)

def get_trial_dir(hparams):
    return get_exp_dir(hparams) / hparams.name / f'version_{hparams.version}'

def get_exp_dir(hparams):
    return ir.LOG_DIR / hparams.dataset_name / hparams.parent_name

def get_exp_name(hparams):
    if hparams.name.lower()[0:4] == 'auto':
        name = f'{hparams.embedding_name}-{hparams.model_size}-{hparams.recurrence_type}-{hparams.loss_fn}-{hparams.random_seed}' + hparams.name[4:]
        if hparams.mixup:
            name = name + '-mixup'
    else:
        return hparams.name
    return name

def parse_model_def(hparams):
    if hasattr(hparams, 'input_repr_model_size'):
        print(hparams.input_repr_model_size)
        hparams.embedding_name, hparams.model_size = hparams.input_repr_model_size
    elif hasattr(hparams, 'input_repr') and hasattr(hparams, 'model_size'):
        pass
    else:
        raise ValueError
    return hparams

def run_task(hparams):
    pl.seed_everything(hparams.random_seed)

    # create custom automatic name if auto
    hparams.name = get_exp_name(hparams)
    # NOTE: this a patch to enable hyperparam search with raytune
    # hparams = parse_model_def(hparams)

    # load the datamodule
    print(f'loading datamodule...')
    dm = DataModule.from_argparse_args(hparams)
    dm.setup()
    hparams.output_dim = len(dm.classlist)

    # define a save dir
    logger_save_dir = get_exp_dir(hparams)
    log_dir = get_trial_dir(hparams)
    hparams.log_dir = str(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    dump_classlist(dm, log_dir)

    # load model
    print(f'loading model...')
    model = Model.from_hparams(hparams)
    
    # build task
    print(f'building task...')
    task = InstrumentDetectionTask.from_hparams(model, dm, hparams)

    # run train fn and get back test results
    print(f'running task')
    task, result = train_instrument_detection_model(task, logger_save_dir=logger_save_dir,  name=hparams.name, version=hparams.version,
                                    gpuid=hparams.gpuid, max_epochs=hparams.max_epochs, random_seed=hparams.random_seed, 
                                    log_dir=hparams.log_dir, test=True)

    # save test results
    ir.utils.data.save_metadata_entry(result, str(log_dir / 'test_result.yaml'), 'json')

    return result

if __name__ == "__main__":
    import argparse
    from instrument_recognition.utils import str2bool

    parser = argparse.ArgumentParser()
    parser.add_argument('--parent_name', type=str, required=True,
        help='name of parent dir where experiment will be stored')

    parser.add_argument('--name', type=str, required=True,
        help='experiment name.')

    parser.add_argument('--version', type=int, required=True, 
        help='experiment version')

    parser.add_argument('--gpuid', type=utils.parser_types.noneint, default=0, 
        help='gpu device number')

    parser.add_argument('--max_epochs', type=int, default=100, 
        help='maximum number of epochs to train for')
        
    parser.add_argument('--random_seed', type=int, default=ir.RANDOM_SEED, 
        help='random seed for experiment')

    parser.add_argument('--test', type=utils.parser_types.str2bool, default=False)

    parser = Model.add_model_specific_args(parser)
    parser = DataModule.add_argparse_args(parser)
    parser = InstrumentDetectionTask.add_model_specific_args(parser)

    hparams = parser.parse_args()

    run_task(hparams)