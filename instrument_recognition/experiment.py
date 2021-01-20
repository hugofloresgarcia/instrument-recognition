""" massive dump of experiment defs
"""
from collections import namedtuple

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


DEFAULT =  {
    'parent_name': None,
    'name': 'auto', 
    'version': 0, 
    'gpuid': 0, 

    'max_epochs': 250, 
    'loss_fn': 'wce', 
    'mixup': False, 
    'mixup_alpha': 0.2,
    'dropout': 0.1, 

    'dataset_name': "mdb-solos", 
    'use_augmented': True, 
    'batch_size': 256, 
    'num_workers': 20,

    'embedding_name': 'openl3-mel256-512-music', 
    'model_size': 'small', 
    'recurrence_type': 'transformer', 
}


from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback

        

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, required=True)
    
    args = parser.parse_args()

    run_experiment(args.name)
