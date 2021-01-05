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


def split(a, n):
    "thank you stack overflow"
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

class InstrumentDetectionTask(pl.LightningModule):

    def __init__(self, model, datamodule, 
                 max_epochs: int = 100,
                 learning_rate: float = 0.0003,
                 loss_fn: str = 'weighted_multiclass_cross_entropy', 
                 mixup: bool = False, mixup_alpha: float = 0.2,
                 log_epoch_metrics: bool = True):
        super().__init__()
        self.save_hyperparameters()
        self.max_epochs = max_epochs
        self.loss_fn = loss_fn
        self.multiclass = 'multiclass' in loss_fn
        self.mixup = mixup
        self.mixup_alpha = mixup_alpha
        self.learning_rate = learning_rate
        self.log_epoch_metrics = log_epoch_metrics

        # datamodules
        self.datamodule = datamodule
        self.classlist = self.datamodule.dataset.classlist

        assert hasattr(self.datamodule.dataset, 'class_weights'), 'dataset is missing self.class_weights'
        self.class_weights = nn.Parameter(
            torch.from_numpy(self.datamodule.dataset.class_weights).float())
        self.class_weights.requires_grad = False

        self.model = model

    @classmethod
    def from_hparams(cls, model, datamodule, hparams):
        obj = cls(model, datamodule,
                    learning_rate=hparams.learning_rate, 
                    loss_fn=hparams.loss_fn, 
                    mixup=hparams.mixup,
                    mixup_alpha=hparams.mixup_alpha, 
                    log_epoch_metrics=hparams.log_epoch_metrics)
        obj.hparams = hparams
        return obj
    
    @classmethod
    def add_argparse_args(cls, parent_parser):
        parser = parent_parser
        parser.add_argument('--learning_rate', type=float, default=0.0003)
        parser.add_argument('--loss_fn', type=str, default='weighted_multiclass_cross_entropy')
        parser.add_argument('--mixup', type=bool, default=False)
        parser.add_argument('--mixup_alpha', type=float, default=0.2)
        parser.add_argument('--log_epoch_metrics', type=bool, default=True)
        return parser

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
        weights = self.class_weights if 'weighted' in self.loss_fn else None
        if 'multiclass_cross_entropy' in self.loss_fn:
            # take the argmax of the y matrix. 
            # y and yhat matrix should be shape (batch, sequence, num_classes)
            # we want y to be shape (batch * sequence) and yhat (batch, sequence, num_classes)
            y = torch.argmax(y, dim=-1).view(-1)
            yhat = yhat.view(-1, yhat.shape[-1])
            loss = F.cross_entropy(yhat, y, weight=weights)
        elif 'binary_cross_entropy' in self.loss_fn:
            y = y.view(-1, y.shape[-1])
            yhat = yhat.view(-1, yhat.shape[-1])
            loss = F.binary_cross_entropy(yhat, y, weight=weights)
        else:
            raise ValueError(f'incorrect loss_fn: {self.loss_fn}')
        
        return loss

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
            raise NotImplementedError
            batch = self.preprocess(batch, train=train)
            X, y_a, y_b, lam = utils.train.mixup_data(batch['X'], batch['y'], alpha=self.mixup_alpha)
            # forward pass through model
            yhat = self.model(X)
            loss = utils.train.mixup_criterion(self.criterion, yhat, y_a, y_b, lam)
            y = y_a if lam > 0.5 else y_b # keep this for traning metrics?
        
        # squash down (sequence, batch, n) to (sequence * batch, n) if multiclass
        yhat = yhat.view(-1, yhat.shape[-1])
        y = y.view(-1, y.shape[-1])

        probits = F.softmax(yhat, dim=1) if self.multiclass else F.sigmoid(yhat)
        yhat = torch.argmax(probits, dim=1, keepdim=False) if self.multiclass else probits

        return dict(loss=loss, y=y.detach(), probits=probits.detach(),
                    yhat=yhat.detach(), X=X.detach())

    def training_step(self, batch, batch_idx):
        result = self._main_step(batch, batch_idx, train=True)

        # update the batch with the result
        batch.update(result)

        # train logging
        self.log('loss/train', result['loss'].detach().cpu(), on_step=True)
        self.log('loss/train-epoch', result['loss'].detach().cpu(), on_step=False, on_epoch=True)

        # pick and log sample audio
        if batch_idx % 250 == 0:
            self.log_random_sample(batch, title='train-sample')
        
        # self.log_sklearn_metrics(batch['yhat'], batch['y'], prefix='train')

        return result['loss']

    def validation_step(self, batch, batch_idx):  
        # get result of forward pass
        result = self._main_step(batch,batch_idx)
        # update the batch with the result 
        batch.update(result)

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
        if self.multiclass:
            assert y.ndim == 1, f'y must NOT be one-hot encoded: {y}'
            # convert to numpy
            probits = probits.detach().numpy()
            y = y.detach().numpy().astype(np.int)

            # compute metrics
            ece = um.ece(y, probits, num_bins=30)
            self.log(f'ECE/{prefix}', ece, prog_bar=True, logger=True, on_epoch=True, on_step=False)

            if prefix == 'test':
                print('computing reliability')
                reliability_fig = um.reliability_diagram(probits, y)

                log_dir = self.log_dir
                confidences, accuracies, xbins = um._reliability_diagram_xy(probits, y)
                confidences = np.array(confidences)
                accuracies = np.array(accuracies)
                xbins = np.array(xbins)
                np.save(os.path.join(log_dir, f'{prefix}-confidences.npy'), confidences)
                np.save(os.path.join(log_dir, f'{prefix}-accuracies.npy'), accuracies)
                np.save(os.path.join(log_dir, f'{prefix}-xbins.npy'), xbins)

                self.logger.experiment.add_figure(f'reliability/{prefix}', reliability_fig, self.global_step)
            print('done :)')
        else: 
            raise NotImplementedError

    def log_multilabel_metrics(self, yhat, y, prefix='val'):
        assert y.ndim == 2 and yhat.ndim == 2
        # transpose to (classes, batch * seq)
        y, yhat = y.permute(1, 0), yhat.permute(1, 0)
        for idx, (tru, probit) in enumerate(zip(y, yhat)):
            label = self.classlist[idx]
            self.log(f'accuracy/{prefix}/{label}', accuracy_score(tru, probit, normalize=True), on_epoch=False)
            self.log(f'precision/{prefix}/{label}', precision_score(tru, probit, average='binary'), on_epoch=False)
            self.log(f'recall/{prefix}/{label}', recall_score(tru, probit,  average='binary'), on_epoch=False)
            self.log(f'fscore/{prefix}{label}', fbeta_score(tru, probit,  average='binary', beta=1), on_epoch=False)
        
    def log_multiclass_metrics(self, yhat, y, prefix='val'):
        self.log(f'accuracy/{prefix}', accuracy_score(y, yhat, normalize=True), on_epoch=False)
        self.log(f'precision/{prefix}', precision_score(y, yhat, average='micro'), on_epoch=False)
        self.log(f'recall/{prefix}', recall_score(y, yhat,  average='micro'), on_epoch=False)
        self.log(f'fscore/{prefix}', fbeta_score(y, yhat,  average='micro', beta=1), on_epoch=False)

    def log_sklearn_metrics(self, yhat, y, prefix='val'):
        if self.multiclass:
            self.log_multiclass_metrics(yhat, y, prefix)
        else:
            self.log_multilabel_metrics(yhat, y, prefix)

    def log_embedding(self, embedding_batch, y, metadata, title):
        assert embedding_batch.ndim == 2
        assert y.ndim == 1
        
        self.logger.experiment.add_embedding(embedding_batch,tag=title,  
                                            metadata=metadata, global_step=self.global_step)

    def log_random_sample(self, batch, title='sample'):
        #TODO: need to add an if multiclass thing here
        if self.multiclass:
            batch = self.batch_detach(batch)
            idx = np.random.randint(0, len(batch['X']))
            pred = self.classlist[batch['yhat'][idx]]
            truth = self.classlist[batch['y'][idx]]
            path_to_audio = batch['path_to_audio'][idx]

            self.logger.experiment.add_text(f'{title}-pred-vs-truth', 
                f'pred: {pred}\n truth:{truth}', 
                self.global_step)

            self.logger.experiment.add_text(f'{title}-path_to_audio/{truth}', 
                                            str(path_to_audio), 
                                            self.global_step)
        else:
            raise NotImplementedError

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
        # TODO: need to add an if multiclass thing here
        # calculate confusion matrices
        yhat = torch.cat([o['yhat'] for o in outputs]).detach().cpu()
        y = torch.cat([o['y'] for o in outputs]).detach().cpu()
        probits =  torch.cat([o['probits'] for o in outputs]).detach().cpu()
        loss = torch.stack([o['loss'] for o in outputs]).detach().cpu()
        
        # print(f'set of val preds: {set(yhat.detach().cpu().numpy())}')
        # print(f'set of val truths: {set(y.detach().cpu().numpy())}')
        if multiclass:
            # CONFUSION MATRIX
            conf_matrix = sklearn.metrics.confusion_matrix(
                y.detach().cpu().numpy(), yhat.detach().cpu().numpy(),
                labels=list(range(len(self.classlist))), normalize=None)
            conf_matrix = np.around(conf_matrix, 3)

            # get plotly images as byte array
            conf_matrix = utils.plot.plot_to_image(utils.plot.plot_confusion_matrix(conf_matrix, self.classlist))

            # CONFUSION MATRIX (NORMALIZED)
            norm_conf_matrix = sklearn.metrics.confusion_matrix(
                y.detach().cpu().numpy(), yhat.detach().cpu().numpy(),
                labels=list(range(len(self.classlist))), normalize='true')
            norm_conf_matrix = np.around(norm_conf_matrix, 2)

            # get plotly images as byte array
            norm_conf_matrix = utils.plot.plot_to_image(utils.plot.plot_confusion_matrix(norm_conf_matrix, self.classlist))

            # log metrics
            self.log(f'accuracy/{prefix}', accuracy_score(y, yhat, normalize=True))
            self.log(f'precision/{prefix}', precision_score(y, yhat, average='micro'))
            self.log(f'recall/{prefix}', recall_score(y, yhat, average='micro'))
            self.log(f'fscore/{prefix}', fbeta_score(y, yhat, average='micro', beta=1))
            self.log(f'fscore_{prefix}', fbeta_score(y, yhat, average='micro', beta=1))

            self.log_uncertainty_metrics(probits, y, prefix)

            self.logger.experiment.add_image(f'conf_matrix/{prefix}', conf_matrix, self.global_step, dataformats='HWC')
            self.logger.experiment.add_image(f'conf_matrix_normalized/{prefix}', norm_conf_matrix, self.global_step, dataformats='HWC')
        else:
            # log metrics
            self.log(f'accuracy/{prefix}', accuracy_score(y, yhat, normalize=True))
            self.log(f'precision/{prefix}', precision_score(y, yhat, average='micro'))
            self.log(f'recall/{prefix}', recall_score(y, yhat, average='micro'))
            self.log(f'fscore/{prefix}', fbeta_score(y, yhat, average='micro', beta=1))
            self.log(f'fscore_{prefix}', fbeta_score(y, yhat, average='micro', beta=1))

            self.logger.experiment.add_text(f'report/{prefix}', 
                classification_report(y, yhat, target_names=self.classlist), self.global_step)

            # reshape to (class, batch) to do per-class scores
            y = y.t()
            yhat = yhat.t()
            for idx, (ytrue, ypred) in enumerate(zip(y, yhat)):
                classname = self.classes[idx]

                self.log(f'accuracy/{classname}-{prefix}/epoch', accuracy_score(ytrue, ypred, normalize=True))
                self.log(f'precision/{classname}-{prefix}/epoch', precision_score(ytrue, ypred))
                self.log(f'recall/{classname}-{prefix}/epoch', recall_score(ytrue, ypred))
                self.log(f'fscore/{classname}-{prefix}/epoch', fbeta_score(ytrue, ypred, beta=1))

        return self.batch_detach(outputs)

def train_instrument_detection_model(task, 
                                    name: str,
                                    version: int,
                                    gpuid: int,
                                    log_dir: str = './test-tubes',
                                    max_epochs: int = 100,
                                    random_seed: int = 20, 
                                    test=False,
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
    os.makedirs(log_dir, exist_ok=True)
    task.log_dir = log_dir
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    best_ckpt = utils.train.get_best_ckpt_path(checkpoint_dir)
    

    # set up checkpoint callbacks
    from pytorch_lightning.callbacks import ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_dir + '/{epoch:02d}-{fscore_val:.2f}', 
        monitor='fscore/val', 
        verbose=True, 
        mode='max',
        save_top_k=3)
    
    if hasattr(task, 'callback_list'):
        callbacks = task.callback_list
    else:
        callbacks = []

    if gpuid is not None:
        if gpuid == -1:
            gpus = -1
            accelerator = 'dp'
        elif isinstance(gpuid, list):
            gpus = gpuid
            accelerator = 'dp'
        else:
            gpus = [gpuid]
            accelerator = None
    else:
        gpus = None
        accelerator = None

    # hardcode some 
    from pytorch_lightning import  Trainer
    trainer = Trainer(
        accelerator=accelerator,
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
        # profiler=True,
        profiler=pl.profiler.SimpleProfiler(
                    output_filename=os.path.join(log_dir, 'profiler-report.txt')), 
        gradient_clip_val=1, 
        deterministic=True,
        num_sanity_val_steps=0, 
        **trainer_kwargs)

    # train, then test
    if not test:
        trainer.fit(task)
        result = trainer.test()
    else:
        result = trainer.test(task)
    return task, result