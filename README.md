

(assuming you have already installed medleydb by adding MEDLEYDB_PATH to your .bashrc)
### step 1: generate chunked dataset

generate 1 second chunks with a hop size of 0.1 at a sample rate of 48000 Hz
(this takes about 4 hours on seethlord on 20 cpus)
```
python generate_dataset.py --path_to_data PATH_TO_DATA --path_to_output PATH_TO_GENERATED_DATA --dataset medleydb --chunk_size 1 --hop_size 0.1 --sr 48000 --num_workers NUM_WORKERS
```

### step 2: add random effects to dtaset

add random_effects to each data sample
```
python augment_dataset.py --path_to_data PATH_TO_GENERATED_DATA --path_to_output PATH_TO_GENERATED_DATA_AUGMENTED --num_workers NUM_WORKERS 
```

### step 3: embed dataset using openl3 

embed dataset using pretrained embedding
```
python embed_dataset.py --path_to_data PATH_TO_AUGMENTED_DATA --path_to_output PATH_TO_EMBEDDING_DATA --batch_size BATCH_SIZE --num_workers NUM_WORKERS
```

