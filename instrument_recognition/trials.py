import datetime
import os

now = datetime.datetime.now().strftime("%d.%m.%Y")
log_dir = os.path.join('logs', f'experiment-MIXUP-{now}')

FIXED = dict(batch_size=256*2, num_workers=20, learning_rate=3e-4,
             weighted_cross_entropy=True, dropout=0.1, random_seed=20, 
             max_epochs=100, log_dir=log_dir,  version=0,  gpuid=-1,
             trainer_kwargs={}) 

# define paths to datamodules 
DM_PATHS = {    
    'SPECTROGRAMS': 
    '/home/hugo/data/mono_music_sed/mdb/SPECTROGRAMS', 

    'EMBEDDINGS-6144':
    '/home/hugo/data/mono_music_sed/mdb/EMBEDDINGS'}


def make_trial(trial_dict):
    """ creates a dict with fixed hyperparams
    and then adds trial_dict on top
    """
    exp_dict = dict(FIXED) 
    exp_dict.update(trial_dict)
    return exp_dict

ensemble_trials = [
    dict(name='openl3-mlp-ensemble-1',
    path_to_data=DM_PATHS['EMBEDDINGS-6144'],
    random_seed=20,
    use_npy= True, 
    model_name='mlp-6144',
    mixup=False,
    mixup_alpha=0),

    dict(name='openl3-mlp-ensemble-2',
    path_to_data=DM_PATHS['EMBEDDINGS-6144'],
    random_seed=21,
    use_npy= True, 
    model_name='mlp-6144',
    mixup=False,
    mixup_alpha=0),

    dict(name='openl3-mlp-ensemble-3',
    path_to_data=DM_PATHS['EMBEDDINGS-6144'],
    random_seed=22,
    use_npy= True, 
    model_name='mlp-6144',
    mixup=False,
    mixup_alpha=0),

    dict(name='openl3-mlp-ensemble-4',
    path_to_data=DM_PATHS['EMBEDDINGS-6144'],
    random_seed=23,
    use_npy= True, 
    model_name='mlp-6144',
    mixup=False,
    mixup_alpha=0),

    dict(name='openl3-mlp-ensemble-5',
    path_to_data=DM_PATHS['EMBEDDINGS-6144'],
    random_seed=24,
    use_npy= True, 
    model_name='mlp-6144',
    mixup=False,
    mixup_alpha=0),

    dict(name='openl3-mlp-ensemble-6',
    path_to_data=DM_PATHS['EMBEDDINGS-6144'],
    random_seed=25,
    use_npy= True, 
    model_name='mlp-6144',
    mixup=False,
    mixup_alpha=0),

    dict(name='openl3-mlp-ensemble-7',
    path_to_data=DM_PATHS['EMBEDDINGS-6144'],
    random_seed=26,
    use_npy= True, 
    model_name='mlp-6144',
    mixup=False,
    mixup_alpha=0),

    dict(name='openl3-mlp-ensemble-8',
    path_to_data=DM_PATHS['EMBEDDINGS-6144'],
    random_seed=27,
    use_npy= True, 
    model_name='mlp-6144',
    mixup=False,
    mixup_alpha=0),
]

mixup_trials = [
    dict(name='openl3-mlp-MIXUP-alpha=0.0',
    path_to_data=DM_PATHS['EMBEDDINGS-6144'],
    use_npy= True, 
    model_name='mlp-6144',
    mixup=False,
    mixup_alpha=0), 
    
    dict(name='openl3-mlp-MIXUP-alpha=0.1',
    path_to_data=DM_PATHS['EMBEDDINGS-6144'],
    use_npy=True, 
    model_name='mlp-6144',
    mixup=True,
    mixup_alpha=0.1),
    

    dict(name='openl3-mlp-MIXUP-alpha=0.2',
    path_to_data=DM_PATHS['EMBEDDINGS-6144'],
    use_npy=True, 
    model_name='mlp-6144',
    mixup=True,
    mixup_alpha=0.2),


    dict(name='openl3-mlp-MIXUP-alpha=0.3',
    path_to_data=DM_PATHS['EMBEDDINGS-6144'],
    use_npy=True, 
    model_name='mlp-6144',
    mixup=True,
    mixup_alpha=0.3),

    dict(name='openl3-mlp-MIXUP-alpha=0.4',
    path_to_data=DM_PATHS['EMBEDDINGS-6144'],
    use_npy=True, 
    model_name='mlp-6144',
    mixup=True,
    mixup_alpha=0.4),

    dict(name='openl3-mlp-MIXUP-alpha=0.5',
    path_to_data=DM_PATHS['EMBEDDINGS-6144'],
    use_npy=True, 
    model_name='mlp-6144',
    mixup=True,
    mixup_alpha=0.5),

    dict(name='openl3-mlp-MIXUP-alpha=0.6',
    path_to_data=DM_PATHS['EMBEDDINGS-6144'],
    use_npy=True, 
    model_name='mlp-6144',
    mixup=True,
    mixup_alpha=0.6),

    dict(name='openl3-mlp-MIXUP-alpha=0.7',
    path_to_data=DM_PATHS['EMBEDDINGS-6144'],
    use_npy=True, 
    model_name='mlp-6144',
    mixup=True,
    mixup_alpha=0.7),

    dict(name='openl3-mlp-MIXUP-alpha=0.7',
    path_to_data=DM_PATHS['EMBEDDINGS-6144'],
    use_npy=True, 
    model_name='mlp-6144',
    mixup=True,
    mixup_alpha=0.7),
]

trials = [
    dict(name='finetune-12.09.2020-mixup', 
     path_to_data=DM_PATHS['SPECTROGRAMS'], 
     batch_size=32*3,
     use_npy=True, 
     model_name='mlp-6144-finetuned|/home/hugo/lab/mono_music_sed/instrument_recognition/weights/openl3-mlp-MIXUP-alpha=0.4', 
     mixup=True, 
     mixup_alpha=0.4,
     trainer_kwargs=dict(accumulate_grad_batches=2),
     max_epochs=20)
]

def get_trial_dict(trial_list):
    return {t['name']: make_trial(t) for t in trial_list}

trials = get_trial_dict(trials)
# ensemble_trials = get_trial_dict(ensemble_trials)
# mixup_trials = get_trial_dict(mixup_trials)
