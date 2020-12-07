import datetime
import os

now = datetime.datetime.now().strftime("%d.%m.%Y")
log_dir = os.path.join('logs', f'experiment-{now}')

FIXED = dict(batch_size=128*3, num_workers=20, learning_rate=3e-4,
             weighted_cross_entropy=True, dropout=0.5, random_seed=20, 
             max_epochs=80, log_dir=log_dir,  version=0,  gpuid=-1,
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


trials = [
    dict(name='openl3-mlp-ERM',
    path_to_data=DM_PATHS['EMBEDDINGS-6144'],
    use_npy= True, 
    model_name='mlp-6144',
    mixup=False,
    mixup_alpha=0), 
    
    dict(name='finetuned-mlp-ERM', 
     path_to_data=DM_PATHS['SPECTROGRAMS'], 
     batch_size=32*3,
     use_npy=False, 
     model_name='openl3mlp-6144', 
     mixup=False, 
     mixup_alpha=0,
     trainer_kwargs=dict(accumulate_grad_batches=8)),
    
    dict(name='baseline-mlp-ERM', 
     path_to_data=DM_PATHS['SPECTROGRAMS'], 
     batch_size=32*3,
     use_npy=False, 
     model_name='baseline-6144', 
     mixup=False, 
     mixup_alpha=0,
     trainer_kwargs=dict(accumulate_grad_batches=8)),
    
    dict(name='openl3-mlp-MIXUP-alpha=0.2',
    path_to_data=DM_PATHS['EMBEDDINGS-6144'],
    use_npy= True, 
    model_name='mlp-6144',
    mixup=True,
    mixup_alpha=0.2),
    
    dict(name='baseline-mlp-MIXUP-alpha=0.2', 
     path_to_data=DM_PATHS['SPECTROGRAMS'], 
     batch_size=32*3,
     use_npy=False, 
     model_name='baseline-6144', 
     mixup=True, 
     mixup_alpha=0.2,
     trainer_kwargs=dict(accumulate_grad_batches=8)),
    
    dict(name='finetuned-mlp-MIXUP-alpha=0.2', 
     path_to_data=DM_PATHS['SPECTROGRAMS'], 
     batch_size=32*3,
     use_npy=False, 
     model_name='openl3mlp-6144', 
     mixup=True, 
     mixup_alpha=0.2,
     trainer_kwargs=dict(accumulate_grad_batches=8)),
    
    # MODELS WITH MIXUP (ALPHA = 0.4)
    dict(name='openl3-mlp-MIXUP-alpha=0.4',
        path_to_data=DM_PATHS['EMBEDDINGS-6144'],
        use_npy= True, 
        model_name='mlp-6144',
        mixup=True,
        mixup_alpha=0.4), 

    dict(name='baseline-mlp-MIXUP-alpha=0.4', 
     path_to_data=DM_PATHS['SPECTROGRAMS'], 
     batch_size=32*3,
     use_npy=False, 
     model_name='baseline-6144', 
     mixup=True, 
     mixup_alpha=0.4,
     trainer_kwargs=dict(accumulate_grad_batches=8)),
    
    dict(name='finetuned-mlp-MIXUP-alpha=0.4', 
     path_to_data=DM_PATHS['SPECTROGRAMS'], 
     batch_size=32*3,
     use_npy=False, 
     model_name='openl3mlp-6144', 
     mixup=True, 
     mixup_alpha=0.4,
     trainer_kwargs=dict(accumulate_grad_batches=8))
]

trials = {t['name']: make_trial(t) for t in trials}