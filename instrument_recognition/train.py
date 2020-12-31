import os

import yaml
import torch
import torch.nn as nn
import pytorch_lightning as pl

import instrument_recognition as ir
import instrument_recognition.utils as utils
from instrument_recognition.task import InstrumentDetectionTask, train_instrument_detection_model
from instrument_recognition.models import Model
from instrument_recognition.datasets import BaseDataModule

def dump_classlist(dm, save_dir):
    # get classlist and number of classes
    classlist = dm.get_classes()
    num_output_units = len(classlist)
    print(f'classlist is: {classlist}')
    with open(os.path.join(save_dir, 'classlist.yaml'), 'w') as f:
        yaml.dump(classlist, f)

def run_task(hparams):
    # load the datamodule
    print(f'loading datamodule...')
    dm = BaseDataModule.from_argparse_args(hparams)

    # define a save dir
    save_dir = os.path.join(ir.LOG_DIR, hparams.name, f'version_{hparams.version}')
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
                                    gpuid=hparams.gpuid, max_epochs=hparams.max_epochs, random_seed=hparams.random_seed, 
                                    log_dir=ir.LOG_DIR, test=hparams.test)

    # save model to torchscript
    utils.train.save_torchscript_model(task.model, os.path.join(save_dir, 'torchscript_model.pt'))

if __name__ == "__main__":
    import argparse
    from instrument_recognition.utils.train import str2bool

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--version', type=int, required=True)

    parser.add_argument('--gpuid', type=utils.train.noneint, default=0)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--test', type=utils.train.str2bool, default=False)

    parser = Model.add_model_specific_args(parser)
    parser = BaseDataModule.add_argparse_args(parser)

    hparams = parser.parse_args()

    run_task(hparams)