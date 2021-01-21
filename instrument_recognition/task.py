"""
the instrument detection task!
most of the experiment code is here. 
"""
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, precision_score, \
                        recall_score, fbeta_score, classification_report
import sklearn
import uncertainty_metrics.numpy as um

import instrument_recognition as ir
import instrument_recognition.utils as utils

def split(a, n):
    "thank you stack overflow"
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

class InstrumentDetectionTask(pl.LightningModule):

    def __init__(self, model, datamodule, 
                 max_epochs: int = 100,
                 learning_rate: float = 0.0003,
                 loss_fn: str = 'wce', 
                 mixup: bool = False, mixup_alpha: float = 0.2, **kwargs):
        super().__init__()
        self.save_hyperparameters('max_epochs', 'learning_rate', 'loss_fn', 'mixup', 'mixup_alpha', *list(kwargs.keys()))
        self.max_epochs = max_epochs
        self.loss_fn = loss_fn
        self.mixup = mixup
        self.mixup_alpha = mixup_alpha
        self.learning_rate = learning_rate

        # datamodules
        self.datamodule = datamodule
        self.classlist = self.datamodule.classlist

        assert hasattr(self.datamodule.dataset, 'class_weights'), 'dataset is missing self.class_weights'
        self.class_weights = nn.Parameter(torch.from_numpy(self.datamodule.dataset.class_weights).float())
        self.class_weights.requires_grad = False

        self.model = model

    @classmethod
    def from_hparams(cls, model, datamodule, hparams):
        obj = cls(model, datamodule, **vars(hparams))
        obj.hparams = hparams
        return obj

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = parent_parser

        parser.add_argument('--learning_rate', type=float, default=0.0003, 
            help='learning rate for training. will be decayed using MultiStepLR')

        parser.add_argument('--loss_fn', type=str, default='wce', 
            help='name of loss function to use: could be wce (weighted cross entropy) or ce (standard cross entropy)')

        parser.add_argument('--mixup', type=ir.utils.str2bool, default=False, 
            help='whether to use mixup training or not')

        parser.add_argument('--mixup_alpha', type=float, default=0.2, 
            help='alpha value for mixup')

        return parser

    def on_train_start(self):
        # log a forward pass
        from torchsummaryX import summary
        from copy import deepcopy
        # get a param count for all models
        sample_input = torch.randn(self.model.input_shape)
        model = deepcopy(self.model)
        model = model.cpu()

        summary(model, sample_input).to_csv(Path(self.log_dir)/'model_summary.csv')

    def batch_detach(self, batch):
        """ detach any tensor in a list and move it to cpu
        """
        for i, item in enumerate(batch):
            if isinstance( item, torch.Tensor):
                batch[i] = item.detach().cpu()
        return batch

    def forward(self, x):
        return self.model(x)
    
    #-------------------------------
    #--------- DATAMODULES ---------
    #-------------------------------
    def train_dataloader(self):
        return self.datamodule.train_dataloader()
    
    def val_dataloader(self):
        return self.datamodule.val_dataloader()

    #-------------------------------
    #---------- TRAINING -----------
    #-------------------------------
    def criterion(self, yhat, y):
        # if the letter "w" is present, the loss function will be weighted
        weights = self.class_weights if 'w' in self.loss_fn else None
        if 'ce' in self.loss_fn:
            # yhat matrix should be shape (batch, sequence, num_classes)
            # we want y to be shape (batch * sequence) and yhat (batch * sequence, num_classes)
            y = y.view(-1)
            yhat = yhat.view(-1, yhat.shape[-1])
            loss = F.cross_entropy(yhat, y, weight=weights)
        else:
            raise ValueError(f'incorrect loss_fn: {self.loss_fn}')
        
        return loss

    def is_seq_batch(self, x):
        return  x.shape[0] < x.shape[1]

    def is_batch_seq(self, x):
        return  x.shape[0] > x.shape[1]

    def batch_seq_swap(self, x):
        """ switch batch and sequence dims """
        return x.permute(1, 0, *list(range(x.ndim))[2:])

    def preprocess(self, batch, train=False):
        # append a copy if our batch size is 1
        if len(batch['X']) == 1:
            batch['X'] = torch.cat([batch['X'], batch['X']], dim=0)
            batch['y'] = torch.cat([batch['y'], batch['y']], dim=0)

        # change batch seq to seq batch
        assert self.is_batch_seq(batch['X']) and self.is_batch_seq(batch['y'])
        batch['X'] = self.batch_seq_swap(batch['X'])
        batch['y'] = self.batch_seq_swap(batch['y'])
        assert self.is_seq_batch(batch['X']) and self.is_seq_batch(batch['y'])

        # take the argmax of y (because y should be onehot)
        assert batch['y'].ndim == 3, f'y should be 3 dimensional one hot but got shape {batch["y"].shape}'
        batch['y'] = torch.argmax(batch['y'], dim=-1, keepdim=False)
        assert batch['y'].ndim == 2

        return batch

    def _main_step(self, batch, batch_idx, train=False):
        if not (self.mixup and train):
            # STANDARD FORWARD PASS
            batch = self.preprocess(batch, train=train)
            X, y = batch['X'], batch['y']
            yhat = self.model(X)
            loss = self.criterion(yhat, y)
        else:
            # MIXUP FORWARD PASS
            batch = self.preprocess(batch, train=train)
            assert self.is_seq_batch(batch['X'])

            #FIXME: make this pretty
            X = batch['X'].reshape(-1, *list(batch['X'].shape)[2:])
            y = batch['y'].reshape(-1, *list(batch['y'].shape)[2:])
            X, y_a, y_b, lam = utils.train.mixup_data(X, y, alpha=self.mixup_alpha, dim=0)
            # forward pass through model
            s, b = batch['X'].shape[:2]
            X = X.view(s, b, *list(batch['X'].shape)[2:])
            y_a = y_a.view(s, b, *list(batch['y'].shape)[2:])
            y_b = y_b.view(s, b, *list(batch['y'].shape)[2:])
            ###

            yhat = self.model(X)
            loss = utils.train.mixup_criterion(self.criterion, yhat, y_a, y_b, lam)
            y = y_a if lam > 0.5 else y_b # keep this for traning metrics?
        
        # squash down (sequence, batch, n) to (sequence * batch, n) if multiclass
        yhat = yhat.view(-1, yhat.shape[-1])
        y = y.view(-1)

        probits = F.softmax(yhat, dim=1)
        yhat = torch.argmax(probits, dim=1, keepdim=False)

        return dict(loss=loss, y=y.detach(), probits=probits.detach(),
                    yhat=yhat.detach(), X=X.detach())

    def training_step(self, batch, batch_idx):
        result = self._main_step(batch, batch_idx, train=True)

        # update the batch with the result
        batch.update(result)
        del result

        # train logging
        self.log('loss/train', batch['loss'].detach().cpu(), on_step=True)
        self.log('loss/train-epoch', batch['loss'].detach().cpu(), on_step=False, on_epoch=True)

        # pick and log sample audio
        if batch_idx % 250 == 0:
            self.log_random_sample(batch, title='train-sample')

        return batch['loss']

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
        self.log_multiclass_metrics(batch['yhat'].detach().cpu(), batch['y'].detach().cpu(), prefix='val')

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
        self.log('loss/test', result['loss'].detach().cpu(), logger=True, prog_bar=True)
        self.log('loss_test', result['loss'].detach().cpu(), on_step=False, on_epoch=True, prog_bar=True)
        self.log_multiclass_metrics(batch['yhat'].detach().cpu(), batch['y'].detach().cpu(), prefix='test')

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
        outputs = self._log_epoch_metrics(outputs, prefix='train')
        return outputs

    def validation_epoch_end(self, outputs):
        outputs = self._log_epoch_metrics(outputs, prefix='val')

    def test_epoch_end(self, outputs):
        outputs = self._log_epoch_metrics(outputs, prefix='test')

    #------------------------------
    #---------- LOGGING -----------
    #------------------------------

    def log_uncertainty_metrics(self, probits, y, prefix='val'):
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
            self.logger.experiment.add_figure(f'reliability/{prefix}', reliability_fig, self.global_step)

            # log_dir = self.log_dir
            # confidences, accuracies, xbins = um._reliability_diagram_xy(probits, y)
            # confidences = np.array(confidences)
            # accuracies = np.array(accuracies)
            # xbins = np.array(xbins)
            # np.save(os.path.join(log_dir, f'{prefix}-confidences.npy'), confidences)
            # np.save(os.path.join(log_dir, f'{prefix}-accuracies.npy'), accuracies)
            # np.save(os.path.join(log_dir, f'{prefix}-xbins.npy'), xbins)

    def log_multilabel_metrics(self, yhat, y, prefix='val'):
        assert y.ndim == 2 and yhat.ndim == 2
        # transpose to (classes, batch * seq)
        y, yhat = y.permute(1, 0), yhat.permute(1, 0)
        for idx, (tru, probit) in enumerate(zip(y, yhat)):
            label = self.classlist[idx]
            probit = np.where(probit > 0.5, np.array(1), np.array(0))
            self.log(f'accuracy/{prefix}/{label}', accuracy_score(tru, probit, normalize=True), on_epoch=False)
            self.log(f'precision/{prefix}/{label}', precision_score(tru, probit, average='binary'), on_epoch=False)
            self.log(f'recall/{prefix}/{label}', recall_score(tru, probit,  average='binary'), on_epoch=False)
            self.log(f'fscore/{prefix}{label}', fbeta_score(tru, probit,  average='binary', beta=1), on_epoch=False)
        
    def log_multiclass_metrics(self, yhat, y, prefix='val'):
        self.log(f'accuracy/{prefix}', accuracy_score(y, yhat, normalize=True), on_epoch=False)
        self.log(f'precision/{prefix}', precision_score(y, yhat, average='weighted'), on_epoch=False)
        self.log(f'recall/{prefix}', recall_score(y, yhat,  average='weighted'), on_epoch=False)
        self.log(f'fscore/{prefix}', fbeta_score(y, yhat,  average='weighted', beta=1), on_epoch=False)

    def log_random_sample(self, batch, title='sample'):
        #TODO: need to add an if multiclass thing here
        batch = self.batch_detach(batch)
        idx = np.random.randint(0, len(batch['X']))
        pred = self.classlist[batch['yhat'][idx]]
        truth = self.classlist[batch['y'][idx]]

        self.logger.experiment.add_text(f'{title}-pred-vs-truth', 
            f'pred: {pred}\n truth:{truth}', 
            self.global_step)

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

    def _epoch_multiclass_report(self, y, yhat, probits, prefix):
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
        self.log(f'precision/{prefix}', precision_score(y, yhat, average='weighted'))
        self.log(f'recall/{prefix}', recall_score(y, yhat, average='weighted'))
        self.log(f'fscore/{prefix}', fbeta_score(y, yhat, average='weighted', beta=1))
        self.log(f'fscore_{prefix}', fbeta_score(y, yhat, average='weighted', beta=1))

        self.log_uncertainty_metrics(probits, y, prefix)

        self.logger.experiment.add_image(f'conf_matrix/{prefix}', conf_matrix, self.global_step, dataformats='HWC')
        self.logger.experiment.add_image(f'conf_matrix_normalized/{prefix}', norm_conf_matrix, self.global_step, dataformats='HWC')

    def _epoch_per_class_report(self, y, yhat, prefix):
        # # log metrics
        # yhat = torch.where(yhat > 0.5, torch.tensor(1), torch.tensor(0))
        # self.log(f'accuracy/{prefix}', accuracy_score(y, yhat, normalize=True))
        # self.log(f'precision/{prefix}', precision_score(y, yhat, average='weighted'))
        # self.log(f'recall/{prefix}', recall_score(y, yhat, average='weighted'))
        # self.log(f'fscore/{prefix}', fbeta_score(y, yhat, average='weighted', beta=1))
        # self.log(f'fscore_{prefix}', fbeta_score(y, yhat, average='weighted', beta=1))

        # CLASSIFICATION REPORTS
        report = classification_report(y, yhat, target_names=self.classlist, output_dict=True)  
        report_fig = utils.plot.plot_to_image(utils.plot.plotly_bce_classification_report(report))
        self.logger.experiment.add_image(f'classification_report/{prefix}', report_fig, self.global_step, dataformats='HWC')

        self.logger.experiment.add_text(f'report/{prefix}', 
            classification_report(y, yhat, target_names=self.classlist), self.global_step)

        # # reshape to (class, batch) to do per-class scores
        # y = y.t()
        # yhat = yhat.t()
        # for idx, (ytrue, ypred) in enumerate(zip(y, yhat)):
        #     classname = self.classlist[idx]

        #     self.log(f'accuracy/{classname}-{prefix}/epoch', accuracy_score(ytrue, ypred, normalize=True))
        #     self.log(f'precision/{classname}-{prefix}/epoch', precision_score(ytrue, ypred))
        #     self.log(f'recall/{classname}-{prefix}/epoch', recall_score(ytrue, ypred))
        #     self.log(f'fscore/{classname}-{prefix}/epoch', fbeta_score(ytrue, ypred, beta=1))

    def _log_epoch_metrics(self, outputs, prefix='val'):
        yhat = torch.cat([o['yhat'] for o in outputs]).detach().cpu()
        y = torch.cat([o['y'] for o in outputs]).detach().cpu()
        probits =  torch.cat([o['probits'] for o in outputs]).detach().cpu()
        loss = torch.stack([o['loss'] for o in outputs]).detach().cpu()
        
        self._epoch_multiclass_report(y, yhat, probits, prefix)
        # self._epoch_per_class_report(y, yhat, prefix)

        return self.batch_detach(outputs)

def train_instrument_detection_model(task, 
                                    logger_save_dir: str,
                                    name: str,
                                    version: int,
                                    gpuid: int,
                                    log_dir: str = './test-tubes',
                                    max_epochs: int = 100,
                                    random_seed: int = 20, 
                                    test=False,
                                    **trainer_kwargs):
    
    # seed everything!!!
    pl.seed_everything(random_seed)

    # set up logger
    from pytorch_lightning.loggers import TestTubeLogger
    logger = TestTubeLogger(
        save_dir=logger_save_dir,
        name=name, 
        version=version)

    if version is None:
        version = logger.version

    # set up logging and checkpoint dirs
    task.log_dir = log_dir
    checkpoint_dir = Path(log_dir) / 'checkpoints'
    best_ckpt = utils.train.get_best_ckpt_path(checkpoint_dir)
    

    # set up checkpoint callbacks
    from pytorch_lightning.callbacks import ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(
        filepath= str(checkpoint_dir  / '{epoch:02d}-{fscore_val:.2f}'), 
        monitor='fscore/val', 
        verbose=True, 
        mode='max',
        save_top_k=3)
    
    if hasattr(task, 'callback_list'):
        callbacks = task.callback_list
    else:
        callbacks = []

    lr_logger = pl.callbacks.LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_logger)

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
    from pytorch_lightning import Trainer
    trainer = Trainer(
        accelerator=accelerator,
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
        profiler=pl.profiler.SimpleProfiler(
                    output_filename=os.path.join(log_dir, 'profiler-report.txt')), 
        gradient_clip_val=1, 
        deterministic=True,
        num_sanity_val_steps=0,
        **trainer_kwargs)

    # train, then test
    if not test:
        trainer.fit(task)
        # result = trainer.test()
        result = None
    else:
        result = trainer.test(task)
    return task, result