import os

import yaml
import torch
import torch.nn as nn
import pytorch_lightning as pl

import instrument_recognition.utils as utils
from instrument_recognition.task import InstrumentDetectionTask, train_instrument_detection_model
from instrument_recognition.models.zoo import load_model
from instrument_recognition.datasets import load_datamodule
from instrument_recognition.trials import trials

def run_trial(exp_dict, model=None, test=False):

    # define a save dir
    save_dir = os.path.join(exp_dict['log_dir'], exp_dict['name'], f'version_{exp_dict["version"]}')
    os.makedirs(save_dir, exist_ok=True)

    # load the datamodule
    print(f'loading datamodule...')
    dm = load_datamodule(path_to_data=exp_dict['path_to_data'], 
                         batch_size=exp_dict['batch_size'], 
                         num_workers=exp_dict['num_workers'],
                         use_npy=exp_dict['use_npy'])
    
    # get classlist and number of classes
    classlist = dm.get_classes()
    num_output_units = len(classlist)
    print(f'classlist is: {classlist}')
    with open(os.path.join(save_dir, 'classlist.yaml'), 'w') as f:
        yaml.dump(classlist, f)

    # load model
    print(f'loading model...')
    if model is None: 
        model = load_model(model_name=exp_dict['model_name'], 
                        output_units=num_output_units, 
                        dropout=exp_dict['dropout'])
    
    # build task
    print(f'building task...')
    task = InstrumentDetectionTask(model, dm, 
                            max_epochs=exp_dict['max_epochs'],
                            learning_rate=exp_dict['learning_rate'], 
                            weighted_cross_entropy=exp_dict['weighted_cross_entropy'], 
                            mixup=exp_dict['mixup'],
                            mixup_alpha=exp_dict['mixup_alpha'], 
                            log_epoch_metrics=True)
    
    # run train fn and get back test results
    print(f'running task')
    task, result = train_instrument_detection_model(task, 
                                    name=exp_dict['name'], 
                                    version=exp_dict['version'], 
                                    gpuid=exp_dict['gpuid'], 
                                    max_epochs=exp_dict['max_epochs'],
                                    random_seed=exp_dict['random_seed'], 
                                    log_dir=exp_dict['log_dir'],
                                    test=test,
                                    **exp_dict['trainer_kwargs'])

    # save exp_dict to yaml file for easy reloading
    with open(os.path.join(save_dir, 'exp_dict.yaml'), 'w') as f:
        yaml.dump(exp_dict, f)

    # save model to torchscript
    utils.train.save_torchscript_model(task.model, os.path.join(save_dir, 'torchscript_model.pt'))

    return task, result

def run_ensemble_trial(exp_dict, num_members=4):
    # generate identical exp dicts with different random seeds
    base_random_seed = exp_dict['random_seed']
    exp_dict['log_dir'] = os.path.join(exp_dict['log_dir'], f'{exp_dict["name"]}-ENSEMBLE') 
    base_log_dir = exp_dict['log_dir']
    print(base_log_dir)
    assert isinstance(base_random_seed, int)
    exp_dict_ensemble = [dict(exp_dict) for m in range(num_members)]
    for idx, e in enumerate(exp_dict_ensemble):
        e['random_seed'] = base_random_seed + idx
        e['log_dir'] = os.path.join(base_log_dir, f'member-{idx}')
    
    # run the task for each model 
    model_list = []
    for exp in exp_dict_ensemble:
        task, result = run_trial(exp)
        task, result = run_trial(exp, test=True)
        model_list.append(task.model)
    
    model = Ensemble(model_list)

    #save and go!
    torch.save(model, os.path.join(base_log_dir, 'modelpkl.pt'))
    torch.save(model.state_dict(), os.path.join(base_log_dir, 'model-statedict.pt'))

    model = torch.load(os.path.join(base_log_dir, 'modelpkl.pt'))
    model.eval()

    exp_dict['name'] += '-ensemble'
    run_trial(exp_dict, model=model)

class Ensemble(pl.LightningModule):

    def __init__(self, model_list):
        super().__init__()
        self.ensemble = nn.ModuleList(model_list)

    def forward(self, x):
        return torch.stack([member(x) for member in self.ensemble]).mean(dim=0)

if __name__ == "__main__":
    import argparse
    from instrument_recognition.utils.train import str2bool

    parser = argparse.ArgumentParser()

    parser.add_argument('--name', required=True, type=str)
    parser.add_argument('--test_only', required=False, type=str2bool, default=False)

    parser.add_argument('--ensemble', required=False, type=str2bool, default=False)
    parser.add_argument('--num_members', required=False, type=int, default=4)

    args = parser.parse_args()

    trial_dict = trials[args.name]
    if not args.ensemble:
        run_trial(trial_dict)
    else:
        run_ensemble_trial(trial_dict, num_members=args.num_members)