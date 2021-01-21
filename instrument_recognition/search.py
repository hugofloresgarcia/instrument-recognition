""" hyperparam search! :)
"""
from functools import partial
import instrument_recognition as ir
import numpy as np
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

DEFAULTS =  {
    'parent_name': 'TUNING',
    'name': 'auto', 
    'version': 0, 
    'gpuid': 0, 
    'random_seed': 42,
    'test': False,

    'max_epochs': 2, 
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

CONFIGS = {
    'ballztothewallz': {
            "learning_rate": tune.loguniform(1e-4, 1e-1),
            "batch_size": tune.choice([32, 64, 128, 256]),
            "input_repr_model_size": tune.choice([('openl3-mel256-512-music', 'tiny'), 
                                    ('openl3-mel256-512-music', 'small'),
                                    ('openl3-mel256-6144-music', 'mid'),
                                    ('openl3-mel256-6144-music', 'huge')]), 
            "use_augmented": tune.choice([True, False]),
            "dropout": tune.choice([0.1, 0.3, 0.5]),
            "recurrence_type": tune.choice(["bilstm", "transformer", "bigru", "none"]),
            "random_seed": tune.choice([1, 13, 23, 42, 440]), 
            "mixup": tune.choice([True, False])
    }, 
}

class Experiment:
    def __init__(self, defaults: dict, config: dict, gpu_fraction: float):
        self.config = config

        self.hparams = argparse.Namespace(**{k: v for k, v in defaults.items() if k not in config})

        self.gpu_fraction = gpu_fraction

def full_name(hparams, keep):
    name = ''
    for k, v in vars(hparams).items():
        if k in keep:
            if isinstance(v, str):
                v = v.replace('openl3', '').replace('music', 'm').replace('transformer', 'xfmr').replace('model_size', 'mdl_sz')\
                    .replace('embedding_name', 'emb').replace('use_augmented', 'aug').replace('mixup', 'mix').replace('random_seed', 'seed')\
                    .replace('recurrence_type', 'rnn')
            if isinstance(k, str):
                k = k.replace('openl3', '').replace('music', 'm').replace('transformer', 'xfmr').replace('model_size', 'mdl_sz')\
                    .replace('embedding_name', 'emb').replace('use_augmented', 'aug').replace('mixup', 'mix').replace('random_seed', 'seed')\
                    .replace('recurrence_type', 'rnn')
            if isinstance(v, float):
                v = np.around(v, 5)
            name = name + f'{k}={v}_'

    name = ''.join(ch for ch in name if ch.isalnum() or ch in ('-', '_', '=', '.'))
    return name

def run_trial(config, **kwargs):
    hparams = argparse.Namespace(**kwargs)
    hparams.__dict__.update(config)

    hparams.name = full_name(hparams, keep=config.keys())

    return ir.train.run_task(hparams)

def run_experiment(exp, num_samples):

    scheduler = ASHAScheduler(
        metric="fscore_val",
        mode="max",
        max_t=exp.hparams.max_epochs,
        grace_period=1, #exp.hparams.max_epochs // 8,
        reduction_factor=2)

    reporter = CLIReporter(
        metric_columns=["loss/train", "fscore_val", "accuracy/val", "training_iteration", 'fscore_test'])

    result = tune.run(
        partial(run_trial, **vars(exp.hparams)),
        name=exp.hparams.parent_name,
        local_dir=str(ir.train.get_exp_dir(exp.hparams)),
        resources_per_trial={"cpu": 1, "gpu": exp.gpu_fraction},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    df = result.results_df
    best_trial = result.get_best_trial(metric="fscore_test", mode="max",)
    df.to_csv(str(ir.train.get_exp_dir(exp.hparams)/'ray-results.csv'))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=1)
    
    args = parser.parse_args()

    exp = Experiment(defaults=DEFAULTS, config=CONFIGS[args.name], gpu_fraction=0.25)

    run_experiment(exp, args.num_samples)
