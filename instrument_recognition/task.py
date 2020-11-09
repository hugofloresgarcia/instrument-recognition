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
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy, Precision, Recall, Fbeta
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score
import sklearn
import matplotlib.pyplot as plt

#TODO: as I get more data modules, 
# the import should just be data_modules.PhilharmoniaDataModule
from instrument_recognition.models.tunedopenl3 import TunedOpenL3
from instrument_recognition.data_modules.philharmonia import PhilharmoniaDataModule
from instrument_recognition.data_modules.slakh import SlakhDataModule
from instrument_recognition.data_modules.mdb import MDBDataModule
from instrument_recognition.utils.plot_utils import plot_to_image, plot_confusion_matrix
from instrument_recognition.utils.train_utils import str2bool, timing, get_best_ckpt_path, Hook, save_torchscript_model, mixup_data, mixup_criterion, noneint
from instrument_recognition.utils import transforms

def load_datamodule(hparams):
    if hparams.dataset.lower() == 'philharmonia':
        datamodule = PhilharmoniaDataModule(
            batch_size=hparams.batch_size, 
            num_workers=hparams.num_dataloader_workers)
    elif hparams.dataset.lower() == 'slakh':
        datamodule = SlakhDataModule(
            batch_size=hparams.batch_size, 
            num_workers=hparams.num_dataloader_workers)
    elif hparams.dataset.lower() == 'medleydb':
        datamodule = MDBDataModule(
            batch_size=hparams.batch_size, 
            num_workers=hparams.num_dataloader_workers)
    else:
        raise ValueError(f'invalid datamodule: {hparams.dataset.lower()}')

    datamodule.setup()
    return datamodule

def load_model(hparams, output_units=None):
    if hparams.model.lower() == 'tunedopenl3':
        #TODO: this output units thing is hacky
        if output_units is not None:
            model = TunedOpenL3(hparams, output_units)
        else:
            model = TunedOpenl3(hparams)

    return model       

class InstrumentDetectionTask(pl.LightningModule):

    def __init__(self, hparams, model, datamodule):
        super().__init__()
        self.hparams = hparams

        # datamodules
        self.datamodule = datamodule
        self.classes = self.datamodule.dataset.classes
        if hasattr(self.datamodule.dataset, 'class_weights') and self.hparams.weighted_cross_entropy:
            self.class_weights = nn.Parameter(
                torch.from_numpy(self.datamodule.dataset.class_weights).float())
            self.class_weights.requires_grad = False
        else:
            self.class_weights = None

        self.model = model
        self.model.log_layers = ['model.' + l for l in self.model.log_layers]

        # register forward hooks for model's logging layers
        self.fwd_hooks = OrderedDict()
        self.register_all_hooks()

        # set up metrics
        # NOTE: pl metrics aren't playing nicely with the gpus :(
        # self.train_accuracy = Accuracy(dist_sync_on_step=True)
        # self.val_accuracy = Accuracy(dist_sync_on_step=True)
        # self.test_accuracy = Accuracy(dist_sync_on_step=True)

        # self.train_precision = Precision(len(self.classes), dist_sync_on_step=True)
        # self.val_precision = Precision(len(self.classes), dist_sync_on_step=True)
        # self.test_precision = Precision(len(self.classes),  dist_sync_on_step=True)

        # self.train_recall = Recall(len(self.classes),  dist_sync_on_step=True)
        # self.val_recall = Recall(len(self.classes),  dist_sync_on_step=True)
        # self.test_recall = Recall(len(self.classes),  dist_sync_on_step=True)

        # self.train_fscore = Fbeta(len(self.classes), dist_sync_on_step=True)
        # self.val_fscore = Fbeta(len(self.classes), dist_sync_on_step=True)
        # self.test_fscore = Fbeta(len(self.classes), dist_sync_on_step=True)

        #  logging
        self.log_dir = None
        
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        # OPTIMIZERS
        parser.add_argument('--learning_rate',  default=1e-4,   type=float)

        # LOGGING
        parser.add_argument('--log_epoch_metrics', default=True, type=str2bool)
        parser.add_argument('--log_train_epoch_metrics', default=False, type=str2bool)

        # MODELS
        parser.add_argument('--dropout',        default=0.5,    type=float) #0.5 dropout is gut
        parser.add_argument('--mixup',          default=False,   type=str2bool)
        parser.add_argument('--mixup_alpha',        default=0.2,    type=float)
        parser.add_argument('--weighted_cross_entropy',     default=True, type=str2bool)

        # AUDIO
        parser.add_argument('--sample_rate',     default=48000, type=int)
        parser.add_argument('--online_transforms', default=False, type=str2bool)

        return parser


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
        if len(audio) == 1:
            audio = torch.cat([batch['X'], batch['X']], dim=0)
            labels = torch.cat([batch['y'], batch['y']], dim=0)

        # batch transforms are now offline
        if train and self.hparams.online_transforms:
            batch['X'] = transforms.random_transform(batch['X'], self.hparams.sample_rate, ['overdrive', 
                                        'reverb', 'pitch', 'stretch'])
        return batch

    def _main_step(self, batch, batch_idx, train=False):
        if not (self.hparams.mixup and train):
            batch = self.preprocess(batch['X'], batch['y'], train=train)
            X, y = batch['X'], batch['y']
            # forward pass through model
            yhat = self.model(X)
            loss = self.criterion(yhat, y)
        else:
            batch = self.preprocess(batch, train=train)
            X, y_a, y_b, lam = mixup_data(batch['X'], batch['y'], alpha=self.hparams.mixup_alpha)
            # forward pass through model
            yhat = self.model(X)
            loss = mixup_criterion(self.criterion, yhat, y_a, y_b, lam)
            y = y_a if lam > 0.5 else y_b # keep this for traning metrics?

        yhat = torch.argmax(F.softmax(yhat, dim=1), dim=1, keepdim=False)

        return dict(loss=loss, y=y.detach().cpu(),
                    yhat=yhat.detach().cpu(), X=X.detach().cpu())

    def training_step(self, batch, batch_idx):
        # get result of forward pass
        result = self._main_step(batch, batch_idx, 
                                mixup=self.hparams.mixup,
                                train=True)

        # update the batch with the result 
        batch.update(result)

        # train logging
        self.log('loss/train', result['loss'], on_step=True)

        # pick and log sample audio
        if batch_idx % 100 == 0:
            self.log_random_example(batch, title='train-sample')
        
        self.log_sklearn_metrics(yhat, y, prefix='train')) 

        return result['loss']

    def validation_step(self, batch, batch_idx):  
        # get result of forward pass
        result = self._main_step(batch,batch_idx)
        # update the batch with the result 
        batch.update(result)

        # log layers defined by model
        for layer in self.model.log_layers:
            self.log_layer_io(layer)

        # pick and log sample audio
        if batch_idx % 100 == 0:
            self.log_random_example(batch, title='train-sample')

        # metric logging
        self.log('loss/val', result['loss'], logger=True, prog_bar=True)
        self.log('loss_val', result['loss'], on_step=False, on_epoch=True, prog_bar=True)
        self.log_sklearn_metrics(yhat, y, prefix='val')

        return result

    def test_step(self, batch, batch_idx):
        # get result of forward pass
        result = self._main_step(batch,batch_idx)
        # update the batch with the result 
        batch.update(result)

        # log layers defined by model
        for layer in self.model.log_layers:
            self.log_layer_io(layer)

        # pick and log sample audio
        if batch_idx % 100 == 0:
            self.log_random_example(batch, title='train-sample')

        # metric logging
        self.log('loss/test', result['loss'], logger=True)     
        self.log_sklearn_metrics(yhat, y, prefix='test')

        return result
    
    # OPTIM
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-5)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min'), 
            'monitor': 'loss_val'
        }
        return [optimizer], [scheduler]

    # EPOCH ENDS
    def train_epoch_end(self, outputs):
        if self.hparams.log_train_epoch_metrics:
            outputs = self._log_epoch_metrics(outputs, prefix='train')
        return outputs

    def validation_epoch_end(self, outputs):
        if self.hparams.log_epoch_metrics:
            outputs = self._log_epoch_metrics(outputs, prefix='val')

    def test_epoch_end(self, outputs):
        if self.hparams.log_epoch_metrics:
            outputs = self._log_epoch_metrics(outputs, prefix='test')

    #-------------------------------
    #-------------------------------
    #---------- LOGGING -----------
    #-------------------------------
    #-------------------------------

    def log_sklearn_metrics(self, yhat, y, prefix='val'):
        y = y.detach().cpu().numpy()
        yhat = yhat.detach().cpu().numpy()
        self.log(f'accuracy/{prefix}', accuracy_score(y, yhat, normalize=True), on_epoch=False)
        self.log(f'precision/{prefix}', precision_score(y, yhat, average='micro'), on_epoch=False)
        self.log(f'recall/{prefix}', recall_score(y, yhat,  average='micro'), on_epoch=False)
        self.log(f'fscore/{prefix}', fbeta_score(y, yhat,  average='micro', beta=1), on_epoch=False)

    def log_random_example(self, batch, title='sample',):
        """
        log a random audio example with predictions and truths!
        """
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
        self.logger.experiment.add_figure(
            layer,
            fig, 
            self.global_step)

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
                self.fwd_hooks[name] = Hook(layer) 
        if not found_layer:
            raise Exception(f'couldnt find at least one of the layers: {layer_names}')
 
    def _log_epoch_metrics(self, outputs, prefix='val'):
        # calculate confusion matrices
        yhat = torch.cat([o['yhat'] for o in outputs]).detach().cpu()
        y = torch.cat([o['y'] for o in outputs]).detach().cpu()
        
        print(f'set of val preds: {set(yhat.detach().cpu().numpy())}')
        print(f'set of val truths: {set(y.detach().cpu().numpy())}')

        for yc in y:
            if yc not in yhat:
                print(f'MODEL DIDNT PREDICT {self.classes[yc]}')

        # CONFUSION MATRIX
        conf_matrix = sklearn.metrics.confusion_matrix(
            y.detach().cpu().numpy(), yhat.detach().cpu().numpy(),
            labels=list(range(len(self.classes))), normalize=None)
        conf_matrix = np.around(conf_matrix, 3)

        # get plotly images as byte array
        conf_matrix = plot_to_image(plot_confusion_matrix(conf_matrix, self.classes))

        # CONFUSION MATRIX (NORMALIZED)
        norm_conf_matrix = sklearn.metrics.confusion_matrix(
            y.detach().cpu().numpy(), yhat.detach().cpu().numpy(),
            labels=list(range(len(self.classes))), normalize='true')
        norm_conf_matrix = np.around(norm_conf_matrix, 2)

        # get plotly images as byte array
        norm_conf_matrix = plot_to_image(plot_confusion_matrix(norm_conf_matrix, self.classes))

        # log images
        self.log(f'accuracy/{prefix}/epoch', accuracy_score(y, yhat, normalize=True))
        self.log(f'precision/{prefix}/epoch', precision_score(y, yhat, average='micro'))
        self.log(f'recall/{prefix}/epoch', recall_score(y, yhat, average='micro'))
        self.log(f'fscore/{prefix}/epoch', fbeta_score(y, yhat, average='micro', beta=1))
        self.logger.experiment.add_image(f'conf_matrix/{prefix}', conf_matrix, self.current_epoch, dataformats='HWC')
        self.logger.experiment.add_image(f'conf_matrix_normalized/{prefix}', norm_conf_matrix, self.current_epoch, dataformats='HWC')

        return outputs

def train_instrument_detection_task(hparams, model):
    from test_tube import Experiment
    
    # see everything!!!
    pl.seed_everything(hparams.random_seed)

    # set the gpu hparam (important for some models)
    hparams.gpus = hparams.gpus or 0

        # set up logger
    logger = pl.loggers.TestTubeLogger(
        save_dir='./test-tubes',
        name=hparams.name, 
        version=hparams.version, 
        create_git_tag=True)

    if hparams.version is None:
        hparams.version = logger.version

    # set up logging and checkpoint dirs
    log_dir = os.path.join(os.getcwd(), 'test-tubes', hparams.name, f'version_{logger.version}')
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    best_ckpt = get_best_ckpt_path(checkpoint_dir)
    
    # add log_dir to hparams for 
    model.log_dir = log_dir

    # set up checkpoint callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir + '/{epoch:02d}-{loss_val:.2f}', 
        monitor='loss/val', 
        verbose=True, 
        save_top_k=3)
    
    if hasattr(model, 'callback_list'):
        callbacks = model.callback_list
    else:
        callbacks = []

    if hparams.gpuid is not None:
        gpus = [hparams.gpuid]
        
    else:
        gpus = None

    # hardcode some 
    trainer = pl.Trainer.from_argparse_args(
        args=hparams,
        default_root_dir=os.path.join(os.getcwd(), 'checkpoints'),
        log_every_n_steps=25,
        callbacks=callbacks,
        checkpoint_callback=checkpoint_callback, 
        logger=logger,
        terminate_on_nan=True,
        resume_from_checkpoint=best_ckpt, 
        weights_summary='full', 
        log_gpu_memory=True, 
        gpus=gpus,
        # profiler=pl.profiler.AdvancedProfiler(), 
        gradient_clip_val=1, 
        )

    if hparams.gpuid is not None:
        hparams.gpus = 1
    else:
        hparams.gpus = 0

    if not hparams.export:
        trainer.fit(model)
        trainer.test()
    
    if hparams.export:
        # model.model.freeze()
        trainer.test(model)
        save_torchscript_model(model.model, 'model.pt')
                                # os.path.join(model.log_dir, 
                                #             f'model_torchscript.pt'))

def get_task_parser():
    import argparse
    parser = argparse.ArgumentParser()

    # by default, the instrument detection task will train
    # TunedOpenL3 
    from instrument_recognition.models.tunedopenl3 import TunedOpenL3
    parser = TunedOpenL3.add_model_specific_args(parser)

    # ------------------
    # | HYPERPARAMETERS |
    # ------------------
    parser = InstrumentDetectionTask.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    # GENERAL
    parser.add_argument('--num_epochs',     default=150,    type=int)
    parser.add_argument('--name',           default='tunedl3-exp', type=str)
    parser.add_argument('--random_seed',    default=42,    type=int)
    parser.add_argument('--verbose',        default=True,  type=str2bool)
    parser.add_argument('--version',        default=None,  type=noneint)
    parser.add_argument('--export',         default=False, type=str2bool)

    # TRAINER
    parser.add_argument('--gpuid',          default=None, type=noneint)

    # DATASETS
    parser.add_argument('--dataset',    default='philharmonia', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_dataloader_workers', default=13, type=int)

    # MODELS
    parser.add_argument('--model',      default='tunedopenl3', type=str)

    return parser

if __name__ == "__main__":
    parser = get_task_parser()

    # parse args
    hparams = parser.parse_args()

    # load datamodule
    datamodule = load_datamodule(hparams)
    print('done w datamodule')

    # load model
    model = load_model(hparams, output_units=len(datamodule.dataset.classes))
    # model = TunedOpenL3(hparams)
    # model = TunedOpenL3.load_from_checkpoint('/home/hugo/lab/mono_music_sed/instrument_recognition/test-tubes/bigpapa-mdb/version_6/checkpoints/epoch=31-loss_val=0.19.ckpt', strict=False, hparams=hparams)

    # load task
    task =  InstrumentDetectionTask(hparams, model, datamodule)

    # run task
    train_instrument_detection_task(hparams, task)