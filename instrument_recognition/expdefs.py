""" I am just literally to lazy to type all these argparse args
"""

class SearchParam:

    def __init__(self, options: list):
        self.options = options

DEFAULTS =  {
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

# a prefix to a venv
PREFIX = "source /home/hugo/lab/venv/bin/activate && cd /home/hugo/lab/instrument-recognition && python -m instrument_recognition.train"

EXPERIMENTS = [
    {
        'parent_name': 'size',
        'name': 'auto', 
        'version': 0, 
        'gpuid': 0, 
        'random_seed': SearchParam(list(range(10))),

        'max_epochs': 10, 
        'loss_fn': 'wce', 
        'mixup': False, 
        'mixup_alpha': 0.2,
        'dropout': 0.1, 

        'dataset_name': "mdb-solos", 
        'use_augmented': True, 
        'batch_size': 256, 
        'num_workers': 20,

        'embedding_name': 'openl3-mel256-512-music', 
        'model_size': SearchParam(['tiny', 'small']), 
        'recurrence_type': 'transformer', 
    },
    {
        'parent_name': 'size',
        'name': 'auto', 
        'version': 0, 
        'gpuid': 0, 
        'random_seed': SearchParam(list(range(10))),

        'max_epochs': 100, 
        'loss_fn': 'wce', 
        'mixup': False, 
        'mixup_alpha': 0.2,
        'dropout': 0.1, 

        'dataset_name': "mdb-solos", 
        'use_augmented': True, 
        'batch_size': 256, 
        'num_workers': 20,

        'embedding_name': 'openl3-mel256-6144-music', 
        'model_size': SearchParam(['mid', 'huge']), 
        'recurrence_type': 'transformer', 
    },

    ####
    {
        'parent_name': 'input_repr',
        'name': 'auto', 
        'version': 0, 
        'gpuid': 1, 
        'random_seed': SearchParam(list(range(10))),

        'max_epochs': 100, 
        'loss_fn': 'wce', 
        'mixup': False, 
        'mixup_alpha': 0.2,
        'dropout': 0.1,

        'dataset_name': "mdb-solos",
        'use_augmented': True, 
        'batch_size': 256,
        'num_workers': 20,

        'embedding_name': SearchParam(['openl3-mel128-512-music', 'openl3-mel256-512-music', 'vggish']),
        'model_size': 'small', 
        'recurrence_type': 'transformer', 
    },

    ####
    {
        'parent_name': 'recurrence',
        'name': 'auto', 
        'version': 0, 
        'gpuid': 2, 
        'random_seed': SearchParam(list(range(10))),

        'max_epochs': 100, 
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
        'recurrence_type': SearchParam(['transformer', 'bilstm', 'bigru', 'none'])
    },

    ####
    {
        'parent_name': 'augmentation',
        'name': 'auto', 
        'version': 0, 
        'gpuid': 2, 
        'random_seed': SearchParam(list(range(10))),

        'max_epochs': 100, 
        'loss_fn': 'wce', 
        'mixup': False, 
        'mixup_alpha': 0.2,
        'dropout': 0.1,

        'dataset_name': "mdb-solos",
        'use_augmented': SearchParam(['true', 'false']),
        'batch_size': 256,
        'num_workers': 20,

        'embedding_name': 'openl3-mel256-512-music',
        'model_size': 'small', 
        'recurrence_type': 'transformer',
    },

    ####
    {
        'parent_name': 'mixup',
        'name': 'auto', 
        'version': 0, 
        'gpuid': 2, 
        'random_seed': SearchParam(list(range(10))),

        'max_epochs': 100, 
        'loss_fn': 'wce', 
        'mixup': SearchParam(['true', 'false']), 
        'mixup_alpha': 0.2,
        'dropout': 0.1,

        'dataset_name': "mdb-solos",
        'use_augmented': 'true',
        'batch_size': 256,
        'num_workers': 20,

        'embedding_name': 'openl3-mel256-512-music',
        'model_size': 'small', 
        'recurrence_type': 'transformer',
    },

]

def expand_dict(d):
    output = []
    for key, param in d.items():
        if isinstance(param, SearchParam):
            for value in param.options:
                new_dict = dict(d)
                new_dict[key] = value
                output.append(new_dict)

    return output

def get_arg_string(exp_dict):
    args = [f'--{k} {v}' for k, v in exp_dict.items()]
    return ' '.join(args)

def print_instance(exp_dict):
    print()
    print(f'# EXPERIMENT')
    print()

    out = ' '.join([PREFIX, get_arg_string(exp_dict)])
    print(out)

    print()
    print()

def print_experiment(name):
    exps = [e for e in EXPERIMENTS if e['parent_name'] == name]
    if len(exps) == 0:
        raise ValueError('wrong exp name')
    
    print('tmux split-window -v "', end=None)
    for idx, exp in enumerate(exps):
        has_search_param = any([isinstance(v, SearchParam) for v in exp.values()])
        
        if has_search_param:
            for subexp in expand_dict(exp):
                print_instance(subexp)
                if idx % 2 == 0:
                    print('" \; split-window -h "')
                else:
                    print('" \; split-window -v "')
                
        else:
            print_instance(exp)
            if idx % 2 == 0:
                print('" \; split-window -h "')
            else:
                print('" \; split-window -v "')



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, required=True)

    name = parser.parse_args().name

    print_experiment(name)
