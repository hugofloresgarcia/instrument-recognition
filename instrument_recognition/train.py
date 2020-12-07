import os

from instrument_recognition.task import InstrumentDetectionTask, train_instrument_detection_model
from instrument_recognition.models.zoo import load_model
from instrument_recognition.datasets import load_datamodule
from instrument_recognition.trials import trials

def run_trial(exp_dict, test_only=False):

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

    # load model
    print(f'loading model...')
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
    result = train_instrument_detection_model(task, 
                                    name=exp_dict['name'], 
                                    version=exp_dict['version'], 
                                    gpuid=exp_dict['gpuid'], 
                                    max_epochs=exp_dict['max_epochs'],
                                    random_seed=exp_dict['random_seed'], 
                                    log_dir=exp_dict['log_dir'],
                                    **exp_dict['trainer_kwargs'])

    # save exp_dict to yaml file for easy reloading
    with open(os.path.join(exp['log_dir'], 'exp_dict.yaml'), 'w') as f:
        yaml.dump(exp_dict, f)
    
    result = result[0]

    return result


if __name__ == "__main__":
    import argparse
    from instrument_recognition.utils.train import str2bool

    parser = argparse.ArgumentParser()

    parser.add_argument('--name', required=True, type=str)
    parser.add_argument('--test_only', required=False, type=str2bool, default=False)

    args = parser.parse_args()

    trial_dict = trials[args.name]
    run_trial(trial_dict, test_only=args.test_only)