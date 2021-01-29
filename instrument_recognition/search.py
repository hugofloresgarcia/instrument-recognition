""" hyperparam search! :)
"""
from functools import partial
import instrument_recognition as ir
import numpy as np
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

SEEDS = [440, 433, 12, 23, 11]
SEEDS = [(1 + n) ** 2 for n in range(10)]
# SEEDS = [s ** 2 for s in SEEDS]

DEFAULTS =  {
    'parent_name': 'DEFAULT',
    'name': 'auto', 
    'version': 0, 
    'gpuid': 0, 
    'random_seed': 440,
    'test': False,

    'max_epochs': 250, 
    'loss_fn': 'wce', 
    'mixup': False, 
    'mixup_alpha': 0.2,
    'dropout': 0.4, 
    'learning_rate': 3e-4,

    'dataset_name': "mdb-solos-train-soundscapes", 
    'use_augmented': False, 
    'batch_size': 256, 
    'num_workers': 20,

    'embedding_name': 'openl3-mel256-6144-music', 
    # NOTE: all of these three should be intertwined?
    'hidden_dim': 1024, 
    'recurrence_type': 'bigru', 
    'recurrence_num_layers': 1,
}

CONFIGS = {
    'input-representation':{
        'dataset_name': tune.grid_search(['mdb-solos', 'mdb-solos-train-soundscapes']),
        'random_seed': tune.grid_search(SEEDS),
        'embedding_name': tune.grid_search(list(ir.models.core.INPUT_DIMS.keys())),
    },
    'transformer-search':{
        'random_seed': tune.grid_search(SEEDS),
        'recurrence_type': tune.grid_search([
            f'transformer-{heads}' for heads in [1, 2, 4, 8, 16, 32, 64]
        ]), 
        'recurrence_num_layers': tune.grid_search([1, 2, 4, 6, 8])
    },
    'bilstm-search':{
        'random_seed': tune.grid_search(SEEDS),
        'recurrence_type': 'bilstm',
        'recurrence_num_layers': tune.grid_search([1, 2, 4, 6, 8])
    },
    'gru-search':{
        'random_seed': tune.grid_search(SEEDS),
        'recurrence_type': 'bigru',
        'recurrence_num_layers': tune.grid_search([1, 2, 4, 6, 8])
    },
    'hidden_dim':{
        'dataset_name': tune.grid_search(['mdb-solos', 'mdb-solos-train-soundscapes']),
        'random_seed': tune.grid_search(SEEDS),
        'hidden_dim': tune.grid_search([128, 256, 512, 1024])
    },
    'augmentation':{
        'dataset_name': tune.grid_search(['mdb-solos', 'mdb-solos-train-soundscapes']),
        'random_seed': tune.grid_search(SEEDS),
        'use_augmented': tune.grid_search([True, False])
    },
    'soundscape':{
        'random_seed': tune.grid_search(SEEDS),
        'dataset_name': tune.grid_search(['mdb-solos', 'mdb-solos-train-soundscapes'])
    },
    'recurrence_type':{
        'dataset_name': tune.grid_search(['mdb-solos', 'mdb-solos-train-soundscapes']),
        'random_seed': tune.grid_search(SEEDS),
        'recurrence_type': tune.grid_search(['bigru', 'bilstm', 'transformer-4', 'none'])
    },
    'enchilada': {
        'dataset_name': 'mdb-solos-train-soundscapes',
        'random_seed': 88, 
        'recurrence_type': 'bilstm', 
        'use_augmented': True, 
    },
    'ballz2dawallz': {
            "learning_rate": tune.loguniform(1e-4, 1e-1),
            "batch_size": tune.choice([32, 64, 128, 256]),
            "input_repr_model_size": tune.choice([('openl3-mel256-512-music', 'tiny'), 
                                    ('openl3-mel256-512-music', 'small'),
                                    ('openl3-mel256-6144-music', 'mid'),
                                    ('openl3-mel256-6144-music', 'huge')]), 
            "use_augmented": tune.choice([True, False]),
            "dropout": tune.uniform(0.0, 0.5),
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

    return ir.train.run_task(hparams, use_ray=True)

def run_experiment(exp, num_samples):

    scheduler = ASHAScheduler(
        metric="fscore_val",
        mode="max",
        max_t=exp.hparams.max_epochs,
        grace_period=exp.hparams.max_epochs // 8,
        reduction_factor=2)

    reporter = CLIReporter(
        metric_columns=["loss/train", "fscore_val", "accuracy/val", "training_iteration", 'fscore_test'])

    result = tune.run(
        partial(run_trial, **vars(exp.hparams)),
        name=exp.hparams.parent_name,
        local_dir=str(ir.train.get_exp_dir(exp.hparams)),
        resources_per_trial={"cpu": 1, "gpu": exp.gpu_fraction},
        config=exp.config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    df = result.results_df
    best_trial = result.get_best_trial(metric="fscore_test", mode="max",)
    df.to_csv(str(ir.train.get_exp_dir(exp.hparams)/'ray-results.csv'))

if __name__ == "__main__":
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--gpu_capacity', type=float, default=1.0)
    
    args = parser.parse_args()

    exp = Experiment(defaults=DEFAULTS, config=CONFIGS[args.name], gpu_fraction= 0.2 / args.gpu_capacity)
    exp.hparams.parent_name = args.name + '-' + datetime.now().strftime("%m.%d.%Y")

    run_experiment(exp, args.num_samples)
