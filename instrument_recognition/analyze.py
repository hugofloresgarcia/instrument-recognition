import os
from pathlib import Path

import pandas as pd
import numpy as np

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

import matplotlib.pyplot as plt
import seaborn as sns
from statannot import add_stat_annotation

import instrument_recognition as ir


def test_and_analyze(exp_name, metric='accuracy/test', variable='recurrence_num_layers',
                    color=None, gpuid=None, random_seed=420):
    # define paths  
    exp_path = ir.LOG_DIR / exp_name
    trial_paths = [exp_path / p / 'version_0' for p in os.listdir(exp_path) if 'test' not in p]
    trial_paths = [p for p in trial_paths if p.exists()]

    test_output_dir = ir.LOG_DIR / 'tests' / exp_name
    os.makedirs(test_output_dir, exist_ok=True)

    print(f'found {len(trial_paths)} trials')
    
    # TEST MODELS AND COLLECT RECORDS
                
    test_records_filepath = test_output_dir / f'{exp_name}.json'

    print(test_records_filepath)
    print(test_records_filepath.exists())
    if not test_records_filepath.exists():
        # run a test loop and collect results
        records = []
        for path in trial_paths:
            try:
                result, hparams = ir.train.test_model_from_checkpoint(path, gpuid=gpuid, 
                                                                      random_seed=random_seed)
                records.append(dict(path=path, hparams=hparams, result=result))
            except AttributeError:
                print(f'reading {path} failed')
                records.append(None)

        # filter out empty records
        records = [r for r in records if r is not None]

        # fix PosixPaths into strings
        for record in records:
            for k in record:
                if isinstance(record[k], Path):
                    record[k] = str(record[k])

        # save a backup of test results
        ir.utils.data.save_metadata_entry(records, str(test_output_dir / exp_name), 'json')
    else:
        print(f'found {test_records_filepath}')
        records = ir.utils.data.load_metadata_entry(str(test_records_filepath))
    
    # expand any dicts of dicts
    expanded_records = []
    for record in records:
        new_record = {}
        for k, v in record.items():
            if isinstance(v, dict):
                new_record.update(v)
            elif isinstance(v, list):
                new_record.update(v[0])
            else:
                new_record[k] = v

        expanded_records.append(new_record)
        
    bigdf = pd.DataFrame(expanded_records)
    
    spvar = 'mdb-solos-train-soundscapes'
    df = bigdf[bigdf['dataset_name'] == spvar]
    print(df)
    uv = list(df[variable].unique())
    uv.sort()

    ax = sns.boxplot(data=df, x=variable, y=metric, hue=color, order=uv, showmeans=True, 
                    meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"10"})
    ax = sns.stripplot(ax=ax, x=variable, y=metric,alpha=0.3, color='black', data=df)
    ax, test_results = add_stat_annotation(ax, data=df, x=variable, y=metric, order=uv, 
                                      box_pairs=[(uv[i], uv[j]) for i in range(len(uv)) \
                                                 for j in range(len(uv)) if i <= j and i != j], 
                                       test="t-test_ind", text_format='star', loc='outside', verbose=2)

    title = f'{spvar}-*-{variable}-*-{metric}'
    ax.set_title(title, pad=len(uv)*40)

    plt.savefig(str(test_output_dir /f'{spvar}-{exp_name}-{metric}-{variable}-{color}.png'.replace('/', '_')), 
                dpi=100, bbox_inches='tight')
    plt.close()
    
#     subplot_field = 'dataset_name'
#     subplot_var = list(bigdf[subplot_field].unique())
#     print(subplot_var)
#     for spvar in subplot_var:
#         df = bigdf[bigdf[subplot_field] == spvar]
#         print(df)
#         uv = list(df[variable].unique())
#         uv.sort()

#         ax = sns.boxplot(data=df, x=variable, y=metric, hue=color, order=uv, showmeans=True, 
#                         meanprops={"marker":"o",
#                            "markerfacecolor":"white", 
#                            "markeredgecolor":"black",
#                           "markersize":"10"})
#         ax = sns.stripplot(ax=ax, x=variable, y=metric,alpha=0.3, color='black', data=df)
#         ax, test_results = add_stat_annotation(ax, data=df, x=variable, y=metric, order=uv, 
#                                           box_pairs=[(uv[i], uv[j]) for i in range(len(uv)) \
#                                                      for j in range(len(uv)) if i <= j and i != j], 
#                                            test="t-test_ind", text_format='star', loc='outside', verbose=2)

#         title = f'{spvar}-*-{variable}-*-{metric}'
#         ax.set_title(title, pad=len(uv)*40)

#         plt.savefig(str(test_output_dir /f'{spvar}-{exp_name}-{metric}-{variable}-{color}.png'.replace('/', '_')), 
#                     dpi=100, bbox_inches='tight')
#         plt.close()
                      

    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--metric", type=str)
    parser.add_argument("--variable", type=str)
    parser.add_argument("--color", type=str)
    parser.add_argument("--exp_name", type=str)
    
    args = parser.parse_args()
    
    test_and_analyze(**vars(args))
    