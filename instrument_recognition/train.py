import os

import yaml
import torch
import torch.nn as nn
import pytorch_lightning as pl

import instrument_recognition as ir
import instrument_recognition.utils as utils
from instrument_recognition.task import InstrumentDetectionTask, train_instrument_detection_model
from instrument_recognition.preprocess import OpenL3Preprocessor
from instrument_recognition.models import Model
from instrument_recognition.datasets import DataModule

def dump_classlist(dm, save_dir):
    # get classlist and number of classes
    classlist = dm.classlist()
    print(f'classlist is: {classlist}')
    with open(os.path.join(save_dir, 'classlist.yaml'), 'w') as f:
        yaml.dump(classlist, f)

def run_task(hparams):
    # load the datamodule
    print(f'loading datamodule...')
    dm = DataModule.from_argparse_args(hparams)
    dm.setup()
    hparams.output_dim = len(dm.classlist())

    # define a save dir
    save_dir = ir.LOG_DIR / hparams.name / f'version_{hparams.version}'
    os.makedirs(save_dir, exist_ok=True)
    dump_classlist(dm, save_dir)

    # load model
    print(f'loading model...')
    model = Model.from_hparams(hparams)
    
    # build task
    print(f'building task...')
    task = InstrumentDetectionTask.from_hparams(model, dm, hparams)

    # run train fn and get back test results
    print(f'running task')
    task, result = train_instrument_detection_model(task, name=hparams.name, version=hparams.version,
                                    gpuid=hparams.gpuid, max_epochs=hparams.max_epochs, random_seed=ir.RANDOM_SEED, 
                                    log_dir=ir.LOG_DIR, test=hparams.test)

    # save model to torchscript
    # utils.train.save_torchscript_model(task.model, os.path.join(save_dir, 'torchscript_model.pt'))

if __name__ == "__main__":
    import argparse
    from instrument_recognition.utils import str2bool

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True,
        help='experiment name')
    parser.add_argument('--version', type=int, required=True)

    parser.add_argument('--gpuid', type=utils.parser_types.noneint, default=0)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--test', type=utils.parser_types.str2bool, default=False)

    parser = Model.add_argparse_args(parser)
    parser = DataModule.add_argparse_args(parser)
    parser = InstrumentDetectionTask.add_argparse_args(parser)

    hparams = parser.parse_args()

    # create custom automatic name if auto
    if hparams.name.lower() == 'auto':
        hparams.name = f'{hparams.dataset_name}-{hparams.embedding_name}-{hparams.model_size}-{hparams.recurrence_type}-{hparams.loss_fn}'

    run_task(hparams)