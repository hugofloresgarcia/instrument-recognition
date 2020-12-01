"""
the instrument detection task!

most of the experiment code is here. 
"""
import argparse
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchvision
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score
import sklearn
import matplotlib.pyplot as plt
import uncertainty_metrics.numpy as um

import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

import instrument_recognition.models as models
import instrument_recognition.utils as utils
import instrument_recognition.datasets.base_dataset as base_dataset

def load_datamodule(path_to_data, batch_size, num_workers, use_embeddings):

    datamodule = base_dataset.BaseDataModule(
        path_to_data=path_to_data,
        batch_size=batch_size, 
        num_workers=num_workers,
        use_embeddings=use_embeddings)
    datamodule.setup()
    
    return datamodule

def load_model(model_name, output_units=None, dropout=0.5):
    """ loads an instrument detection model
    options: tunedopenl3, openl3mlp6144-finetuned, openl3mlp-512,
             openl3mlp-6144, mlp-512, mlp-6144
    """
    if model_name == 'tunedopenl3':
        from instrument_recognition.models.tunedopenl3 import TunedOpenL3
        model = TunedOpenL3(output_units, dropout, sr=48000, use_kapre=False)
    
    elif model_name == 'baseline-512':
        from instrument_recognition.models.tunedopenl3 import OpenL3MLP
        model = OpenL3MLP(embedding_size=512, dropout=dropout, num_output_units=output_units, 
                          sr=48000, pretrained=False)

    elif model_name == 'baseline-6144':
        from instrument_recognition.models.tunedopenl3 import OpenL3MLP
        model = OpenL3MLP(embedding_size=6144, dropout=dropout, num_output_units=output_units, 
                          sr=48000, pretrained=False)

    elif model_name == 'openl3mlp6144-finetuned':
        from instrument_recognition.models.torchopenl3 import OpenL3Embedding
        from instrument_recognition.models.mlp import MLP6144
        model = nn.Sequential(OrderedDict([
            ('openl3', OpenL3Embedding(128, 6144)),
            ('mlp', MLP6144(0.5, 39)), 
        ]))
        mlp_state_d  = torch.load('/home/hugo/lab/mono_music_sed/instrument_recognition/weights/mlp-6144-mixup')
        print('LOADING MLP STATE DICT')
        print(mlp_state_d.keys())
        model.mlp.load_state_dict(mlp_state_d)
        
    elif model_name == 'openl3mlp-512':
        from instrument_recognition.models.tunedopenl3 import OpenL3MLP
        model = OpenL3MLP(embedding_size=512, 
                          dropout=dropout,
                          num_output_units=output_units)

    elif model_name == 'openl3mlp-6144':
        from instrument_recognition.models.tunedopenl3 import OpenL3MLP
        model = OpenL3MLP(embedding_size=6144, 
                          dropout=dropout,
                          num_output_units=output_units)

    elif model_name == 'mlp-512':
        from instrument_recognition.models.mlp import MLP512
        model = MLP512(dropout, output_units)  

    elif model_name == 'mlp-6144':
        from instrument_recognition.models.mlp import MLP6144
        model = MLP6144(dropout, output_units)

    else:
        raise ValueError(f"couldnt find model name: {model_name}")

    return model

class InstrumentDetectionTask(pl.LightningModule):

    def __init__(self, model, datamodule, 
                 max_epochs,
                 learning_rate=0.0003,
                 weighted_cross_entropy=True, 
                 mixup=False, mixup_alpha=0.2,
                 log_epoch_metrics=True, 
                 ):
        super().__init__()
        self.max_epochs = max_epochs
        self.weighted_cross_entropy = weighted_cross_entropy
        self.mixup = mixup
        self.mixup_alpha = mixup_alpha
        self.learning_rate = learning_rate
        self.log_epoch_metrics = log_epoch_metrics

        # datamodules
        self.datamodule = datamodule
        self.classes = self.datamodule.dataset.classes

        # do weighted cross entropy? 
        if self.weighted_cross_entropy:
            assert hasattr(self.datamodule.dataset, 'class_weights'), 'dataset is missing self.class_weights'
            self.class_weights = nn.Parameter(
                torch.from_numpy(self.datamodule.dataset.class_weights).float())
            self.class_weights.requires_grad = False
        else:
            self.class_weights = None

        self.model = model
        # self.model.log_layers = ['model.' + l for l in self.model.log_layers]

        # register forward hooks for model's logging layers
        # self.fwd_hooks = OrderedDict()
        # self.register_all_hooks()

    def batch_detach(self, batch):
        for i, item in enumerate(batch):
            if isinstance( item, torch.Tensor):
                batch[i] = item.detach().cpu()
        return batch

    def forward(self, x):
        return self.model(x)
    #-------------------------------
    #-------------------------------
    #--------- DATAMODULES ---------
    #-------------------------------
    #-------------------------------
    def train_dataloader(self):
        return self.datamodule.train_dataloader()
    
    def val_dataloader(self):
        return self.datamodule.val_dataloader()

    def test_dataloader(self):
        return self.datamodule.test_dataloader()

    #-------------------------------
    #-------------------------------
    #---------- TRAINING -----------
    #-------------------------------
    #-------------------------------
    
    def criterion(self, yhat, y):
        return F.cross_entropy(yhat, y, weight=self.class_weights)

    def preprocess(self, batch, train=False):
        if len(batch['X']) == 1:
            batch['X'] = torch.cat([batch['X'], batch['X']], dim=0)
            batch['y'] = torch.cat([batch['y'], batch['y']], dim=0)

        return batch

    def _main_step(self, batch, batch_idx, train=False):
        if not (self.mixup and train):
            batch = self.preprocess(batch, train=train)
            X, y = batch['X'], batch['y']
            # forward pass through model
            yhat = self.model(X)
            loss = self.criterion(yhat, y)
        else:
            batch = self.preprocess(batch, train=train)
            X, y_a, y_b, lam = utils.train.mixup_data(batch['X'], batch['y'], alpha=self.mixup_alpha)
            # forward pass through model
            yhat = self.model(X)
            loss = utils.train.mixup_criterion(self.criterion, yhat, y_a, y_b, lam)
            y = y_a if lam > 0.5 else y_b # keep this for traning metrics?

        probits = F.softmax(yhat, dim=1)
        yhat = torch.argmax(probits, dim=1, keepdim=False)

        return dict(loss=loss, y=y.detach(), probits=probits.detach(),
                    yhat=yhat.detach(), X=X.detach())

    def training_step(self, batch, batch_idx):

        # get result of forward pass
        result = self._main_step(batch, batch_idx, train=True)

        # update the batch with the result
        batch.update(result)

        # train logging
        self.log('loss/train', result['loss'].detach().cpu(), on_step=True)
        self.log('loss/train-epoch', result['loss'].detach().cpu(), on_step=False, on_epoch=True)

        # pick and log sample audio
        if batch_idx % 250 == 0:
            self.log_random_sample(batch, title='train-sample')
            # print(f'done:)')
        
        self.log_sklearn_metrics(batch['yhat'], batch['y'], prefix='train')

        return result['loss']

    def validation_step(self, batch, batch_idx):  
        # get result of forward pass
        result = self._main_step(batch,batch_idx)
        # update the batch with the result 
        batch.update(result)

        # # log layers defined by model
        # for layer in self.model.log_layers:
        #     self.log_layer_io(layer)

        # pick and log sample audio
        if batch_idx % 100 == 0:
            self.log_random_sample(batch, title='val-sample')

        # metric logging
        self.log('loss/val', result['loss'].detach().cpu(), logger=True, prog_bar=True)
        self.log('loss_val', result['loss'].detach().cpu(), on_step=False, on_epoch=True, prog_bar=True)
        self.log_sklearn_metrics(batch['yhat'], batch['y'], prefix='val')

        # result['loss'] = result['loss'].detach().cpu()
        return result

    def test_step(self, batch, batch_idx):
        # get result of forward pass
        result = self._main_step(batch,batch_idx)
        # update the batch with the result 
        batch.update(result)

        # # log layers defined by model
        # for layer in self.model.log_layers:
        #     self.log_layer_io(layer)

        # pick and log sample audio
        if batch_idx % 100 == 0:
            self.log_random_sample(batch, title='test-sample')

        # metric logging
        self.log('loss/test', result['loss'].detach().cpu(), logger=True)     
        self.log_sklearn_metrics(batch['yhat'], batch['y'], prefix='test')

        return result
    
    # OPTIM
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            # add this lambda so it doesn't crash if part of the model is frozen
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate, 
            weight_decay=1e-5)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                              milestones=[0.5 * self.max_epochs,
                                                                          0.75 * self.max_epochs], 
                                                              gamma=0.1)}
        return [optimizer], [scheduler]

    # EPOCH ENDS
    def train_epoch_end(self, outputs):
        if self.log_epoch_metrics:
            outputs = self._log_epoch_metrics(outputs, prefix='train')
        return outputs

    def validation_epoch_end(self, outputs):
        if self.log_epoch_metrics:
            outputs = self._log_epoch_metrics(outputs, prefix='val')

    def test_epoch_end(self, outputs):
        if self.log_epoch_metrics:
            outputs = self._log_epoch_metrics(outputs, prefix='test')

    #-------------------------------
    #-------------------------------
    #---------- LOGGING -----------
    #-------------------------------
    #-------------------`------------

    def log_uncertainty_metrics(self, probits, y, prefix='val'):
        assert y.ndim == 1, f'y must NOT be one-hot encoded: {y}'
        # convert to numpy
        probits = probits.detach().numpy()
        y = y.detach().numpy().astype(np.int)

        # compute metrics
        ece = um.ece(y, probits, num_bins=30)
        self.log(f'ECE/{prefix}', ece, prog_bar=True, logger=True, on_epoch=True, on_step=False)

        # if prefix == 'test':
        #     print('computing reliability')
        #     max_sample_size = 10000
        #     if len(y) > max_sample_size:
        #         indices = np.random.choice(np.arange(len(y)), max_sample_size)
        #         y = np.take(y, indices, axis=0)
        #         probits = np.take(y, indices, axis=0)
        #     reliability_fig = um.reliability_diagram(probits, y)
        #     self.logger.experiment.add_figure(f'reliability/{prefix}', reliability_fig, self.global_step)
        #     print('done :)')

    def log_sklearn_metrics(self, yhat, y, prefix='val'):
        y = y.detach().cpu().numpy()
        yhat = yhat.detach().cpu().numpy()
        self.log(f'accuracy/{prefix}', accuracy_score(y, yhat, normalize=True), on_epoch=False)
        self.log(f'precision/{prefix}', precision_score(y, yhat, average='micro'), on_epoch=False)
        self.log(f'recall/{prefix}', recall_score(y, yhat,  average='micro'), on_epoch=False)
        self.log(f'fscore/{prefix}', fbeta_score(y, yhat,  average='micro', beta=1), on_epoch=False)

    def log_embedding(self, embedding_batch, y, metadata, title):
        assert embedding_batch.ndim == 2
        assert y.ndim == 1
        
        self.logger.experiment.add_embedding(embedding_batch,tag=title,  metadata=metadata, global_step=self.global_step)

    def log_random_sample(self, batch, title='sample'):
        batch = self.batch_detach(batch)
        idx = np.random.randint(0, len(batch['X']))
        pred = self.classes[batch['yhat'][idx]]
        truth = self.classes[batch['y'][idx]]
        path_to_audio = batch['path_to_audio'][idx]

        self.logger.experiment.add_text(f'{title}-pred-vs-truth', 
            f'pred: {pred}\n truth:{truth}', 
            self.global_step)

        self.logger.experiment.add_text(f'{title}-path_to_audio/{truth}', 
                                        str(path_to_audio), 
                                        self.global_step)

    def log_random_example_deprecated(self, batch, title='sample',):
        """
        log a random audio example with predictions and truths!
        """
        raise NotImplementedError 
        audio = batch['X']
        yhat = batch['yhat']
        y = batch['y']
        path_to_audio = batch['path_to_audio']

        idx = np.random.randint(0, len(audio))
        pred = self.classes[yhat[idx]]
        truth = self.classes[y[idx]]

        self.logger.experiment.add_audio(f'{title}-audio/{truth}', 
            audio[idx], self.global_step, self.hparams.sample_rate)

        self.logger.experiment.add_text(f'{title}-predVtruth', 
            f'pred: {pred}\n truth:{truth}', 
            self.global_step)
        
        self.logger.experiment.add_text(f'{title}-probits', 
            f'pred: {yhat[idx].numpy()} \n truth: {y[idx].numpy()}', 
            self.global_step)
        
        self.logger.experiment.add_text(f'{title}-audiopath', str(path_to_audio[idx]), self.global_step)

    def log_example(self, audio, sr, yhat, y, title='sample'):
        """note: example must be a SINGLE example. 
        remove batch dimension b4 calling this
        """
        self.logger.experiment.add_audio(title, 
                                        audio.detach().cpu(), 
                                        self.global_step, 
                                        sr)
        pred = self.classes[int(yhat.detach().cpu())]
        truth = self.classes[int(y.detach().cpu())]
        self.logger.experiment.add_text(f'{title}', 
            f'pred: {pred}\n truth:{truth}', 
            self.global_step)

    def log_layer_io(self, layer):
        """
        log a random example in the output of a layer 
        and the batch's shape and stats on a text entry
        """

        # get the layer
        batch = self.fwd_hooks[layer].output.detach().cpu()
        idx = torch.randint(0, len(batch), (1,))
        o = batch[idx][0]
        
        # if the layer is 2D, display as image
        # if the layer is 1D, log as a heatmap
        if o.ndim == 1:
            # a sad attempt at a square img
            o = o.view(32, -1)
        elif o.ndim == 3:
            # o = torchvision.utils.make_grid(   
            #     o.unsqueeze(1), nrow=int(np.sqrt(o.shape[0])),
            #     normalize=True)
            o = o[0]
        # elif o.ndim > 3:
        #     raise Exception(f'cannot log tensor with ndim {o.ndim+1}')
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.imshow(o)
        

        # log
        self.logger.experiment.add_figure(layer, fig, self.global_step)

        plt.close(fig)

        # log layer statistics
        mean = batch.mean()
        std = batch.std()
        minimum = torch.min(batch)
        maximum = torch.max(batch)

        layer_stats = f'{layer:<40}\n\tmean:{mean}\n\std:{std}\n\min:{minimum}\n\max:{maximum}'

        self.logger.experiment.add_text(layer, layer_stats, self.global_step)

    def register_all_hooks(self):
        layer_names = self.model.log_layers
        if layer_names is []:
            return

        found_layer = False
        for name, layer in self.named_modules():
            if name in layer_names:
                found_layer = True
                self.fwd_hooks[name] = utils.train.Hook(layer) 
        if not found_layer:
            raise Exception(f'couldnt find at least one of the layers: {layer_names}')
 
    def _log_epoch_metrics(self, outputs, prefix='val'):
        # calculate confusion matrices
        yhat = torch.cat([o['yhat'] for o in outputs]).detach().cpu()
        y = torch.cat([o['y'] for o in outputs]).detach().cpu()
        probits =  torch.cat([o['probits'] for o in outputs]).detach().cpu()
        loss = torch.cat([o['loss'] for o in outputs]).detach().cpu()
        
        # print(f'set of val preds: {set(yhat.detach().cpu().numpy())}')
        # print(f'set of val truths: {set(y.detach().cpu().numpy())}')

        # for yc in y:
        #     if yc not in yhat:
        #         print(f'MODEL DIDNT PREDICT {self.classes[yc]}')

        # CONFUSION MATRIX
        conf_matrix = sklearn.metrics.confusion_matrix(
            y.detach().cpu().numpy(), yhat.detach().cpu().numpy(),
            labels=list(range(len(self.classes))), normalize=None)
        conf_matrix = np.around(conf_matrix, 3)

        # get plotly images as byte array
        conf_matrix = utils.plot.plot_to_image(utils.plot.plot_confusion_matrix(conf_matrix, self.classes))

        # CONFUSION MATRIX (NORMALIZED)
        norm_conf_matrix = sklearn.metrics.confusion_matrix(
            y.detach().cpu().numpy(), yhat.detach().cpu().numpy(),
            labels=list(range(len(self.classes))), normalize='true')
        norm_conf_matrix = np.around(norm_conf_matrix, 2)

        # get plotly images as byte array
        norm_conf_matrix = utils.plot.plot_to_image(utils.plot.plot_confusion_matrix(norm_conf_matrix, self.classes))

        # log images
        self.log(f'accuracy/{prefix}/epoch', accuracy_score(y, yhat, normalize=True))
        self.log(f'precision/{prefix}/epoch', precision_score(y, yhat, average='micro'))
        self.log(f'recall/{prefix}/epoch', recall_score(y, yhat, average='micro'))
        self.log(f'fscore/{prefix}/epoch', fbeta_score(y, yhat, average='micro', beta=1))

        self.log_uncertainty_metrics(probits, y, prefix)

        self.logger.experiment.add_pr_curve(f'pr_curve/{prefix}', labels=y, predictions=yhat, global_step=self.global_step)
        self.logger.experiment.add_image(f'conf_matrix/{prefix}', conf_matrix, self.global_step, dataformats='HWC')
        self.logger.experiment.add_image(f'conf_matrix_normalized/{prefix}', norm_conf_matrix, self.global_step, dataformats='HWC')

        return self.batch_detach(outputs)

def train_instrument_detection_model(model, 
                                    name: str,
                                    version: int,
                                    gpuid: int,
                                    log_dir: str = './test-tubes',
                                    max_epochs: int = 100,
                                    random_seed: int = 20, 
                                    **trainer_kwargs):
    from test_tube import Experiment
    
    # seed everything!!!
    pl.seed_everything(random_seed)


    # set up logger
    from pytorch_lightning.loggers import TestTubeLogger
    logger = TestTubeLogger(
        save_dir=log_dir,
        name=name, 
        version=version, 
        create_git_tag=True)

    if version is None:
        version = logger.version

    # set up logging and checkpoint dirs
    log_dir = os.path.join(log_dir, name, f'version_{version}')
    # os.makedirs(log_dir, exist_ok=False)
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    best_ckpt = utils.train.get_best_ckpt_path(checkpoint_dir)
    

    # set up checkpoint callbacks
    from pytorch_lightning.callbacks import ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_dir + '/{epoch:02d}-{loss_val:.2f}', 
        monitor='loss/val', 
        verbose=True, 
        save_top_k=3)
    
    if hasattr(model, 'callback_list'):
        callbacks = model.callback_list
    else:
        callbacks = []

    if gpuid is not None:
        if gpuid == -1:
            gpus = -1
        else:
            gpus = [gpuid]
    else:
        gpus = None

    # hardcode some 
    from pytorch_lightning import  Trainer
    trainer = Trainer(
        accelerator='dp',
        # distributed_backend='dp',
        log_every_n_steps=100,
        max_epochs=max_epochs,
        callbacks=callbacks,
        checkpoint_callback=checkpoint_callback, 
        logger=logger,
        terminate_on_nan=True,
        resume_from_checkpoint=best_ckpt, 
        weights_summary='full', 
        log_gpu_memory=True, 
        gpus=gpus,
        profiler=True,
        # profiler=pl.profiler.SimpleProfiler(
        #             output_filename=os.path.join(log_dir, 'profiler-report.txt')), 
        gradient_clip_val=1, 
        deterministic=True,
        num_sanity_val_steps=0, 
        **trainer_kwargs)

    # train, then test
    trainer.fit(model)
    test_log = trainer.test()
    return test_log
