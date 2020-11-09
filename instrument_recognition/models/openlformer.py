import torch
import torchaudio
import pytorch_lightning as pl
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torchvision


from .openl3_torch import OpenL3Mel128
from .base_training_module import BaseTrainingModule
from .data_modules import *
from .xfmr import PositionalEncoding, MultiheadAttention

from .utils.plot_utils import plot_to_image, plot_confusion_matrix

import math

import sklearn

def softmax_pool(x, dim=1):
    return (x * x).sum(dim = dim) / x.sum(dim = dim)

def OpenMicTransform( x, old_sr, new_sr, target_len):
        # downmix if needed
        if x.shape[0] > 1:
            x = x.mean(dim=0, keepdim=False)
        else:
            x = x.squeeze(0)

        x = torchaudio.transforms.Resample(old_sr, new_sr)(x)

        # zero pad to meet target len
        x = audio_utils.zero_pad(x, target_len)
        x = x.view(1, target_len)

        return x

class OpenLFormer(BaseTrainingModule):

    def __init__(self, hparams):
        super().__init__(hparams)
        self.hparams = hparams
        self.datamodule = OpenMicDataModule(
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_dataloader_workers
        )
        self.datamodule.setup()
        self.classes = self.datamodule.dataset.classes

        self.sr=48000
        self.openl3_res = 29
        self.seq_len = self.hparams.seq_len

        # take 10 seconds of audio in
        self.openl3 = OpenL3Mel128(input_shape=(1, self.sr * self.seq_len),
                             maxpool_kernel=(16,24), maxpool_stride=(4, 8))

        # freeze and unfreeze later
        self.openl3.freeze_up_to('maxpool')
        self.openl3.is_frozen = True

        # self.pos_encoding = PositionalEncoding(d_model=512,
        #                                     dropout=self.hparams.dropout) 
        
        # self.loss_weights = torch.tensor([1/f for n, f in self.datamodule.train_data.get_class_frequencies()]) 
        # self.loss_weights = self.loss_weights / torch.norm(self.loss_weights)

        # if self.hparams.gpus > 0:
        #     self.loss_weights = self.loss_weights.cuda()

        # have a weak prediction come out of the encoder
        # have strong predictions come out of the decoder

        self.fc_seq1 = nn.Sequential(nn.Linear(512, 128), nn.ReLU())
    

        self.mha = MultiheadAttention(8, 128)

        # # # transformer! 
        # self.transformer = nn.Transformer(
        #     d_model=512,
        #     nhead=8, 
        #     num_encoder_layers=4,
        #     num_decoder_layers=4, 
        #     dim_feedforward=128, 
        #     dropout=self.hparams.dropout, 
        #     activation='relu', 
        # )

        self.fc_seq = nn.Sequential(
            nn.Linear(128, 20), 
            nn.Sigmoid(),
        )

    
    def forward(self, x):
        # x should be the openl3 melspec already

        x = self.openl3(x)

        # concat freq and channel dimensions
        x = x.view(-1, 512, self.openl3_res)
        # reshape to (sequence, batch, embedding)
        x = x.permute(2, 0, 1) 
        x = self.fc_seq1(x)
        # positional encoding
        # x = x + self.pos_encoding(x)

        # forward pass thru transform
        # x = self.transformer(x, x)

        # go back to (batch, sequence, embedding)
        x = x.permute(1, 0, 2)

        # pass thru multihead att
        x = self.mha(x)

        # get our attention weights
        # att_w = self.att_seq(x)
        # average over sequence dim
        # att_w = att_w / torch.sum(att_w, dim=1, keepdim=True)

        # cool! now, forward pass thru our fc seq
        # to get segment level predictions
        x = self.fc_seq(x)

        return x
    
    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument('--label_threshold', default=0.5,   type=float)
        parser.add_argument('--dropout',        default=0.1,    type=float)
        parser.add_argument('--seq_len',        default=10,     type=int)
        return parser

    def preprocess(self, batch):
        X = []
        Y = []
        for e in batch:
            x = e['audio']
            x = OpenMicTransform(x, e['sr'], self.sr, self.seq_len * self.sr)
            X.append(x)

            y = torch.tensor(e['onehot'])
            Y.append(y)
        X = torch.stack(X)
        Y = torch.stack(Y).float()

        if self.hparams.gpus > 0:
            Y = Y.cuda()

        return dict(X=X, Y=Y)
        
    def criterion(self, y_pred, y_true):
        return F.binary_cross_entropy(y_pred, y_true)

    def training_step(self, batch, batch_idx):
        data = self.preprocess(batch)
        audio = data['X']
        y_true_weak = data['Y']

        # get a melspec of the 10s of audio
        spec = self.openl3.melspec(audio, gpu=self.hparams.gpus)

        if self.openl3.is_frozen:
            self.openl3.eval()
        else:
            self.openl3.train()

        # forward pass thru model
        y_strong = self(spec)

        # do softmax to get a strong label (over the sequence dimension)
        # softmax pooling!
        # pred_weak = torch.sum(y_strong * attn_w, dim=1, keepdim=False)
        prob_weak = softmax_pool(y_strong, dim=1)
        
        # nice! now, let's run our loss
        prob_weak.clamp_(min = 1e-7, max = 1 - 1e-7)

        one = torch.tensor(1).float().cuda() if self.hparams.gpus > 0 else torch.tensor(1).float()
        zero = torch.tensor(0).float().cuda() if self.hparams.gpus > 0 else torch.tensor(0).float()

        pred_strong = torch.where(y_strong > self.hparams.label_threshold, one, zero)
        loss = self.criterion(prob_weak, y_true_weak) #+ self.criterion(attn_w, y_true_strong)
        
        # RESULTS
        result = pl.TrainResult(minimize=loss)
        result.log('train_loss', loss, on_step=True)

        return result

    def validation_step(self, batch, batch_idx):
        data = self.preprocess(batch)
        audio = data['X']
        y_true_weak = data['Y']

        # get a melspec of the 10s of audio
        spec = self.openl3.melspec(audio, gpu=self.hparams.gpus)

        if self.openl3.is_frozen:
            self.openl3.eval()
        else:
            self.openl3.train()

        # forward pass thru model
        y_strong = self(spec)

        # do softmax to get a strong label (over the sequence dimension)
        # softmax pooling!
        # pred_weak = torch.sum(y_strong * attn_w, dim=1, keepdim=False)
        prob_weak = softmax_pool(y_strong, dim=1)
        
        # nice! now, let's run our loss
        prob_weak.clamp_(min = 1e-7, max = 1 - 1e-7)

        one = torch.tensor(1).float().cuda() if self.hparams.gpus > 0 else torch.tensor(1).float()
        zero = torch.tensor(0).float().cuda() if self.hparams.gpus > 0 else torch.tensor(0).float()
        pred_strong = torch.where(y_strong > self.hparams.label_threshold, one, zero)
        loss = self.criterion(prob_weak, y_true_weak) #+ self.criterion(attn_w, y_true_strong)

        pred_weak = torch.where(prob_weak > self.hparams.label_threshold, one, zero).detach().cpu()
        y_true_weak = y_true_weak.detach().cpu()
        pred_strong = pred_strong.detach().cpu()

        global_step = 1000 * self.current_epoch + batch_idx

        # self.logger.experiment.add_audio(f'val_audio', audio[0], global_step)
        
        # self.logger.experiment.add_text(f'strong pred dim', str(pred_strong.shape), global_step)
        self.logger.experiment.add_text(f'strong prob', str(y_strong), global_step)
        self.logger.experiment.add_text(f'weak prob', str(prob_weak), global_step)

        self.logger.experiment.add_image(
            f'weak_probs', 
            torch.tensor(prob_weak.detach().cpu()), 
            global_step,
            dataformats='HW', 
        )

        self.logger.experiment.add_image(
            f'weak_pred', 
            torch.tensor(pred_weak.detach().cpu()), 
            global_step,
            dataformats='HW', 
        )

        self.logger.experiment.add_image(
            f'weak_truf', 
            torch.tensor(y_true_weak.detach().cpu()), 
            global_step,
            dataformats='HW', 
        )

        self.logger.experiment.add_images(
            f'strong_predictions',
            torch.tensor(pred_strong).view(-1, pred_strong.shape[-1]),
            global_step,
            dataformats='WH',
        )

        self.logger.experiment.add_images(
            f'strong_probs',
            torch.tensor(y_strong).view(-1, y_strong.shape[-1]),
            global_step,
            dataformats='WH',
        )

        # self.logger.experiment.add_images(
        #     f'attn_matrix', 
        #     torch.tensor(attn_w).unsqueeze(1), 
        #     global_step,
        #     dataformats='NCHW', 
        # )

        pred_weak = torch.where(pred_weak > self.hparams.label_threshold, torch.tensor(1), torch.tensor(0))

        result = pl.EvalResult(checkpoint_on=loss, early_stop_on=loss)

        result.pred_weak = pred_weak
        result.y_true_weak = y_true_weak
        
        result.log('val_loss',  loss)

        return result

    def _log_epoch_metrics(self, outputs, prefix='val'):
        yhat = outputs.pred_weak
        y = outputs.y_true_weak

        # super()._log_epoch_metrics(outputs, prefix)

        norm_conf_matrix = sklearn.metrics.multilabel_confusion_matrix(
            yhat.detach().numpy(), y.detach().numpy(), 
            labels=list(range(len(self.classes))))
        norm_conf_matrix = np.around(norm_conf_matrix, 3)

        # get plotly images as byte array
        # norm_conf_matrix = plot_to_image(plot_confusion_matrix(norm_conf_matrix, self.classes))

        # log images
        for label, cm in zip(self.classes, norm_conf_matrix):
            cm = plot_to_image(plot_confusion_matrix(cm, [label, f'not {label}']))
            self.logger.experiment.add_image(f'{prefix}_{label}_conf_matrix_normalized', 
                cm, self.current_epoch, dataformats='HWC')

        outputs.yhat = torch.tensor([0])
        outputs.y = torch.tensor([0])

        accuracy = torch.tensor(sklearn.metrics.accuracy_score(y, yhat))
        f1 = torch.tensor(sklearn.metrics.f1_score(y, yhat, average='macro'))

        self.logger.experiment.add_scalar(f'{prefix}_accuracy', accuracy, self.current_epoch)
        self.logger.experiment.add_scalar(f'{prefix}_f1', f1, self.current_epoch)

        return outputs

class OpenL3Attn(BaseTrainingModule):

    def __init__(self, hparams):
        super().__init__(hparams)
        self.hparams = hparams
        self.datamodule = OpenMicDataModule(
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_dataloader_workers
        )
        self.datamodule.setup()
        self.classes = self.datamodule.dataset.classes

        self.sr=48000
        self.openl3_res = 29
        self.seq_len = self.hparams.seq_len

        # take 10 seconds of audio in
        self.openl3 = OpenL3Mel128(input_shape=(1, self.sr * self.seq_len),
                             maxpool_kernel=(16,24), maxpool_stride=(4, 8))

        # freeze and unfreeze later
        self.openl3.freeze_up_to('maxpool')
        self.openl3.unfreeze_starting_at('conv2d_4')


        self.key = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 128, (1, 1)),  
            nn.LeakyReLU(), 
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 20, (1, 1)), 
            nn.Sigmoid()
        )

        self.val = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 128, (1, 1)),  
            nn.LeakyReLU(), 
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 20, (1, 1)), 
            nn.Sigmoid()
        )

    
    def forward(self, x):
        # x should be the openl3 melspec already
        x = self.openl3(x)

        # concat freq and channel dimensions
        x = x.view(-1, 512, self.openl3_res, 1)

        # get our attention weights
        p = self.key(x)
        a = self.val(x)
        # shape is (batch, embedding, time, 1)

        a = a.squeeze(-1).permute(0, 2, 1)
        p = p.squeeze(-1).permute(0, 2, 1)

        return p, a
    
    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument('--label_threshold', default=0.5,   type=float)
        parser.add_argument('--dropout',        default=0.1,    type=float)
        parser.add_argument('--seq_len',        default=10,     type=int)
        return parser

    def preprocess(self, batch):
        X = []
        Y = []
        for e in batch:
            x = e['audio']
            x = OpenMicTransform(x, e['sr'], self.sr, self.seq_len * self.sr)
            X.append(x)

            y = torch.tensor(e['onehot'])
            Y.append(y)
        X = torch.stack(X)
        Y = torch.stack(Y).float()

        if self.hparams.gpus > 0:
            Y = Y.cuda()

        return dict(X=X, Y=Y)
        
    def criterion(self, y_pred, y_true):
        return F.binary_cross_entropy(y_pred, y_true)

    def training_step(self, batch, batch_idx):
        data = self.preprocess(batch)
        audio = data['X']
        y_true_weak = data['Y']

        # get a melspec of the 10s of audio
        spec = self.openl3.melspec(audio, gpu=self.hparams.gpus)

        # forward pass thru model
        a, p = self(spec)
        y_strong = p

        # do softmax to get a strong label (over the sequence dimension)
        e = 1e-7
        a = torch.clamp(a, e, 1. - e)

        a = a / torch.sum(a, dim=1, keepdim=True)
        x = torch.sum(a * p, dim=1)
        prob_weak = torch.sum(a * p, dim=1)
        
        # nice! now, let's run our loss
        prob_weak.clamp_(min = 1e-7, max = 1 - 1e-7)

        one = torch.tensor(1).float().cuda() if self.hparams.gpus > 0 else torch.tensor(1).float()
        zero = torch.tensor(0).float().cuda() if self.hparams.gpus > 0 else torch.tensor(0).float()

        pred_strong = torch.where(y_strong > self.hparams.label_threshold, one, zero)
        loss = self.criterion(prob_weak, y_true_weak) #+ self.criterion(attn_w, y_true_strong)
        
        # RESULTS
        result = pl.TrainResult(minimize=loss)
        result.log('train_loss', loss, on_step=True)

        return result

    def validation_step(self, batch, batch_idx):
        data = self.preprocess(batch)
        audio = data['X']
        y_true_weak = data['Y']

        # get a melspec of the 10s of audio
        spec = self.openl3.melspec(audio, gpu=self.hparams.gpus)

        # forward pass thru model
        a, p = self(spec)
        y_strong = p

        # do softmax to get a strong label (over the sequence dimension)
        e = 1e-7
        a = torch.clamp(a, e, 1. - e)

        a = a / torch.sum(a, dim=1, keepdim=True)
        x = torch.sum(a * p, dim=1)
        prob_weak = torch.sum(a * p, dim=1)
        
        # nice! now, let's run our loss
        prob_weak.clamp_(min = 1e-7, max = 1 - 1e-7)

        one = torch.tensor(1).float().cuda() if self.hparams.gpus > 0 else torch.tensor(1).float()
        zero = torch.tensor(0).float().cuda() if self.hparams.gpus > 0 else torch.tensor(0).float()
        pred_strong = torch.where(y_strong > self.hparams.label_threshold, one, zero)
        loss = self.criterion(prob_weak, y_true_weak) #+ self.criterion(attn_w, y_true_strong)

        pred_weak = torch.where(prob_weak > self.hparams.label_threshold, one, zero).detach().cpu()
        y_true_weak = y_true_weak.detach().cpu()
        pred_strong = pred_strong.detach().cpu()

        global_step = 1000 * self.current_epoch + batch_idx

        
        self.logger.experiment.add_text(f'strong prob', str(y_strong), global_step)
        self.logger.experiment.add_text(f'weak prob', str(prob_weak), global_step)

        self.logger.experiment.add_image(
            f'weak_probs', 
            torch.tensor(prob_weak.detach().cpu()), 
            global_step,
            dataformats='HW', 
        )

        self.logger.experiment.add_image(
            f'weak_pred', 
            torch.tensor(pred_weak.detach().cpu()), 
            global_step,
            dataformats='HW', 
        )

        self.logger.experiment.add_image(
            f'weak_truf', 
            torch.tensor(y_true_weak.detach().cpu()), 
            global_step,
            dataformats='HW', 
        )

        self.logger.experiment.add_images(
            f'strong_predictions',
            torch.tensor(pred_strong).view(-1, pred_strong.shape[-1]),
            global_step,
            dataformats='WH',
        )

        self.logger.experiment.add_images(
            f'strong_probs',
            torch.tensor(y_strong).reshape(-1, y_strong.shape[-1]),
            global_step,
            dataformats='WH',
        )

        # self.logger.experiment.add_images(
        #     f'attn_matrix', 
        #     torch.tensor(attn_w).unsqueeze(1), 
        #     global_step,
        #     dataformats='NCHW', 
        # )

        pred_weak = torch.where(pred_weak > self.hparams.label_threshold, torch.tensor(1), torch.tensor(0))

        result = pl.EvalResult(checkpoint_on=loss, early_stop_on=loss)

        result.pred_weak = pred_weak
        result.y_true_weak = y_true_weak
        
        result.log('val_loss',  loss)

        return result

    def _log_epoch_metrics(self, outputs, prefix='val'):
        yhat = outputs.pred_weak
        y = outputs.y_true_weak

        # super()._log_epoch_metrics(outputs, prefix)

        norm_conf_matrix = sklearn.metrics.multilabel_confusion_matrix(
            yhat.detach().numpy(), y.detach().numpy(), 
            labels=list(range(len(self.classes))))
        norm_conf_matrix = np.around(norm_conf_matrix, 3)

        # get plotly images as byte array
        # norm_conf_matrix = plot_to_image(plot_confusion_matrix(norm_conf_matrix, self.classes))

        # log images
        for label, cm in zip(self.classes, norm_conf_matrix):
            cm = plot_to_image(plot_confusion_matrix(cm, [label, f'not {label}']))
            self.logger.experiment.add_image(f'{prefix}_{label}_conf_matrix_normalized', 
                cm, self.current_epoch, dataformats='HWC')

        outputs.yhat = torch.tensor([0])
        outputs.y = torch.tensor([0])

        accuracy = torch.tensor(sklearn.metrics.accuracy_score(y, yhat))
        f1 = torch.tensor(sklearn.metrics.f1_score(y, yhat, average='macro'))

        self.logger.experiment.add_scalar(f'{prefix}_accuracy', accuracy, self.current_epoch)
        self.logger.experiment.add_scalar(f'{prefix}_f1', f1, self.current_epoch)

        return outputs

class VGGishAttn(BaseTrainingModule):

    def __init__(self, hparams):
        super().__init__(hparams)

        from .vggish_openmic import VGGishOpenMicDataModule

        self.hparams = hparams
        self.datamodule = VGGishOpenMicDataModule(
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_dataloader_workers
        )
        self.datamodule.setup()

        temp_dm = OpenMicDataModule(
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_dataloader_workers
        )
        temp_dm.setup()
        self.classes = temp_dm.dataset.classes


        self.seq_len = self.hparams.seq_len

        embedding_layers = [
            nn.Sequential(
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 128, (1, 1), bias=False),
                nn.ReLU(),
                nn.Dropout2d(p=self.hparams.dropout)
            ) for i in range(3)
        ]

        embedding_layers.append(nn.BatchNorm2d(128))

        self.emb = nn.Sequential(*embedding_layers)

        self.key = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 20, (1, 1)), 
            nn.Sigmoid()
        )

        self.val = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 20, (1, 1)), 
            nn.Sigmoid()
        )

    
    def forward(self, x):
        # x should be the openl3 melspec already

        x = self.emb(x) + x

        p = self.key(x)
        a = self.val(x)
        
        # shape is (batch, embedding, time, 1)
        a = a.squeeze(-1).permute(0, 2, 1)
        p = p.squeeze(-1).permute(0, 2, 1)

        return p, a
    
    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument('--label_threshold', default=0.5,   type=float)
        parser.add_argument('--dropout',        default=0.6,    type=float)
        parser.add_argument('--seq_len',        default=10,     type=int)
        return parser

    def preprocess(self, batch):
        X = []
        Y = []
        for e in batch:
            x = e['audio']
            x = OpenMicTransform(x, e['sr'], self.sr, self.seq_len * self.sr)
            X.append(x)

            y = torch.tensor(e['onehot'])
            Y.append(y)
        X = torch.stack(X)
        Y = torch.stack(Y).float()

        if self.hparams.gpus > 0:
            Y = Y.cuda()

        return dict(X=X, Y=Y)
        
    def criterion(self, y_pred, y_true):
        return F.binary_cross_entropy(y_pred, y_true)

    def training_step(self, batch, batch_idx):
        X, Y, Y_true, Y_mask = batch
        X = X.permute(0, 2, 1).unsqueeze(-1)

        y_true_weak = Y_true

        # forward pass thru model
        a, p = self(X)
        y_strong = p

        # do softmax to get a strong label (over the sequence dimension)
        e = 1e-7
        a = torch.clamp(a, e, 1. - e)

        a = a / torch.sum(a, dim=1, keepdim=True)
        prob_weak = torch.sum(a * p, dim=1)
        prob_weak = F.hardtanh(prob_weak, 0., 1.)
        
        # nice! now, let's run our loss
        prob_weak.clamp_(min = 1e-7, max = 1 - 1e-7)

        loss = self.criterion(prob_weak, y_true_weak) #+ self.criterion(attn_w, y_true_strong)
        
        # RESULTS
        result = pl.TrainResult(minimize=loss)
        result.log('train_loss', loss, on_step=True)

        return result

    def validation_step(self, batch, batch_idx):
        X, Y, Y_true, Y_mask = batch
        X = X.permute(0, 2, 1).unsqueeze(-1)

        y_true_weak = Y_true

        # forward pass thru model
        a, p = self(X)
        y_strong = p

        # do softmax to get a strong label (over the sequence dimension)
        e = 1e-7
        a = torch.clamp(a, e, 1. - e)

        a = a / torch.sum(a, dim=1, keepdim=True)
        prob_weak = torch.sum(a * p, dim=1)
        prob_weak = F.hardtanh(prob_weak, 0., 1.)
        
        # nice! now, let's run our loss
        prob_weak.clamp_(min = 1e-7, max = 1 - 1e-7)

        one = torch.tensor(1).float().cuda() if self.hparams.gpus > 0 else torch.tensor(1).float()
        zero = torch.tensor(0).float().cuda() if self.hparams.gpus > 0 else torch.tensor(0).float()
        pred_strong = torch.where(y_strong > self.hparams.label_threshold, one, zero)
        loss = self.criterion(prob_weak, y_true_weak) #+ self.criterion(attn_w, y_true_strong)

        pred_weak = torch.where(prob_weak > self.hparams.label_threshold, one, zero).detach().cpu()
        y_true_weak = y_true_weak.detach().cpu()
        pred_strong = pred_strong.detach().cpu()

        global_step = 1000 * self.current_epoch + batch_idx

        
        self.logger.experiment.add_text(f'strong prob', str(y_strong), global_step)
        self.logger.experiment.add_text(f'weak prob', str(prob_weak), global_step)

        self.logger.experiment.add_image(
            f'weak_probs', 
            torch.tensor(prob_weak.detach().cpu()), 
            global_step,
            dataformats='HW', 
        )

        self.logger.experiment.add_image(
            f'weak_pred', 
            torch.tensor(pred_weak.detach().cpu()), 
            global_step,
            dataformats='HW', 
        )

        self.logger.experiment.add_image(
            f'weak_truf', 
            torch.tensor(y_true_weak.detach().cpu()), 
            global_step,
            dataformats='HW', 
        )

        self.logger.experiment.add_images(
            f'strong_predictions',
            torch.tensor(pred_strong).reshape(-1, pred_strong.shape[-1]),
            global_step,
            dataformats='WH',
        )

        self.logger.experiment.add_images(
            f'strong_probs',
            torch.tensor(y_strong).reshape(-1, y_strong.shape[-1]),
            global_step,
            dataformats='WH',
        )

        self.logger.experiment.add_images(
            f'attn_matrix', 
            torch.tensor(a).reshape(-1, a.shape[-1]), 
            global_step,
            dataformats='WH', 
        )

        result = pl.EvalResult(checkpoint_on=loss, early_stop_on=loss)

        result.pred_weak = pred_weak
        result.y_true_weak = y_true_weak
        
        result.log('val_loss',  loss)

        return result

    def _log_epoch_metrics(self, outputs, prefix='val'):
        yhat = outputs.pred_weak
        y = outputs.y_true_weak

        # super()._log_epoch_metrics(outputs, prefix)

        norm_conf_matrix = sklearn.metrics.multilabel_confusion_matrix(
            yhat.detach().numpy(), y.detach().numpy(), 
            labels=list(range(len(self.classes))))
        norm_conf_matrix = np.around(norm_conf_matrix, 3)

        # get plotly images as byte array
        # norm_conf_matrix = plot_to_image(plot_confusion_matrix(norm_conf_matrix, self.classes))

        # log images
        for label, cm in zip(self.classes, norm_conf_matrix):
            cm = plot_to_image(plot_confusion_matrix(cm, [label, f'not {label}']))
            self.logger.experiment.add_image(f'{prefix}_{label}_conf_matrix_normalized', 
                cm, self.current_epoch, dataformats='HWC')

        outputs.yhat = torch.tensor([0])
        outputs.y = torch.tensor([0])

        accuracy = torch.tensor(sklearn.metrics.accuracy_score(y, yhat))
        f1 = torch.tensor(sklearn.metrics.f1_score(y, yhat, average='macro'))

        self.logger.experiment.add_scalar(f'{prefix}_accuracy', accuracy, self.current_epoch)
        self.logger.experiment.add_scalar(f'{prefix}_f1', f1, self.current_epoch)

        return outputs

class VGGishMHA(BaseTrainingModule):

    def __init__(self, hparams):
        super().__init__(hparams)

        from .vggish_openmic import VGGishOpenMicDataModule

        self.hparams = hparams
        self.datamodule = VGGishOpenMicDataModule(
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_dataloader_workers
        )
        self.datamodule.setup()

        temp_dm = OpenMicDataModule(
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_dataloader_workers
        )
        temp_dm.setup()
        self.classes = temp_dm.dataset.classes


        self.seq_len = self.hparams.seq_len

        embedding_layers = [
            nn.Sequential(
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 128, (1, 1), bias=False),
                nn.ReLU(),
                nn.Dropout2d(p=self.hparams.dropout)
            ) for i in range(3)
        ]

        self.emb = nn.Sequential(*embedding_layers)

        self.mha = MultiheadAttention(8, 128)

        self.fc_seq = nn.Sequential(
            nn.Linear(128, 20), 
            nn.Sigmoid(),
        )

    
    def forward(self, x):
        # x should be the openl3 melspec already

        x = self.emb(x) + x
        x = x.squeeze(-1).permute(0, 2, 1)

        x = self.mha(x)

        x = self.fc_seq(x)

        return x
    
    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument('--label_threshold', default=0.5,   type=float)
        parser.add_argument('--dropout',        default=0.1,    type=float)
        parser.add_argument('--seq_len',        default=10,     type=int)
        return parser

    def preprocess(self, batch):
        X = []
        Y = []
        for e in batch:
            x = e['audio']
            x = OpenMicTransform(x, e['sr'], self.sr, self.seq_len * self.sr)
            X.append(x)

            y = torch.tensor(e['onehot'])
            Y.append(y)
        X = torch.stack(X)
        Y = torch.stack(Y).float()

        if self.hparams.gpus > 0:
            Y = Y.cuda()

        return dict(X=X, Y=Y)
        
    def criterion(self, y_pred, y_true):
        return F.binary_cross_entropy(y_pred, y_true)

    def training_step(self, batch, batch_idx):
        X, Y, Y_true, Y_mask = batch
        X = X.permute(0, 2, 1).unsqueeze(-1)

        y_true_weak = Y_true

        # forward pass thru model
        y_strong = self(X)

        # do softmax to get a strong label (over the sequence dimension)
        prob_weak = softmax_pool(y_strong, dim=1)
        
        # nice! now, let's run our loss
        prob_weak.clamp_(min = 1e-7, max = 1 - 1e-7)

        loss = self.criterion(prob_weak, y_true_weak) #+ self.criterion(attn_w, y_true_strong)
        
        # RESULTS
        result = pl.TrainResult(minimize=loss)
        result.log('train_loss', loss, on_step=True)

        return result

    def validation_step(self, batch, batch_idx):
        X, Y, Y_true, Y_mask = batch
        X = X.permute(0, 2, 1).unsqueeze(-1)

        y_true_weak = Y_true

        # forward pass thru model
        y_strong = self(X)

        # do softmax to get a strong label (over the sequence dimension)
        prob_weak = softmax_pool(y_strong, dim=1)
        
        # nice! now, let's run our loss
        prob_weak.clamp_(min = 1e-7, max = 1 - 1e-7)

        one = torch.tensor(1).float().cuda() if self.hparams.gpus > 0 else torch.tensor(1).float()
        zero = torch.tensor(0).float().cuda() if self.hparams.gpus > 0 else torch.tensor(0).float()
        pred_strong = torch.where(y_strong > self.hparams.label_threshold, one, zero)
        loss = self.criterion(prob_weak, y_true_weak) #+ self.criterion(attn_w, y_true_strong)

        pred_weak = torch.where(prob_weak > self.hparams.label_threshold, one, zero).detach().cpu()
        y_true_weak = y_true_weak.detach().cpu()
        pred_strong = pred_strong.detach().cpu()

        global_step = 1000 * self.current_epoch + batch_idx

        
        self.logger.experiment.add_text(f'strong prob', str(y_strong), global_step)
        self.logger.experiment.add_text(f'weak prob', str(prob_weak), global_step)

        self.logger.experiment.add_image(
            f'weak_probs', 
            torch.tensor(prob_weak.detach().cpu()), 
            global_step,
            dataformats='HW', 
        )

        self.logger.experiment.add_image(
            f'weak_pred', 
            torch.tensor(pred_weak.detach().cpu()), 
            global_step,
            dataformats='HW', 
        )

        self.logger.experiment.add_image(
            f'weak_truf', 
            torch.tensor(y_true_weak.detach().cpu()), 
            global_step,
            dataformats='HW', 
        )

        self.logger.experiment.add_images(
            f'strong_predictions',
            torch.tensor(pred_strong).reshape(-1, pred_strong.shape[-1]),
            global_step,
            dataformats='WH',
        )

        self.logger.experiment.add_images(
            f'strong_probs',
            torch.tensor(y_strong).reshape(-1, y_strong.shape[-1]),
            global_step,
            dataformats='WH',
        )

        result = pl.EvalResult(checkpoint_on=loss, early_stop_on=loss)

        result.pred_weak = pred_weak
        result.y_true_weak = y_true_weak
        
        result.log('val_loss',  loss)

        return result

    def _log_epoch_metrics(self, outputs, prefix='val'):
        yhat = outputs.pred_weak
        y = outputs.y_true_weak

        # super()._log_epoch_metrics(outputs, prefix)

        norm_conf_matrix = sklearn.metrics.multilabel_confusion_matrix(
            yhat.detach().numpy(), y.detach().numpy(), 
            labels=list(range(len(self.classes))))
        norm_conf_matrix = np.around(norm_conf_matrix, 3)

        # get plotly images as byte array
        # norm_conf_matrix = plot_to_image(plot_confusion_matrix(norm_conf_matrix, self.classes))

        # log images
        for label, cm in zip(self.classes, norm_conf_matrix):
            cm = plot_to_image(plot_confusion_matrix(cm, [label, f'not {label}']))
            self.logger.experiment.add_image(f'{prefix}_{label}_conf_matrix_normalized', 
                cm, self.current_epoch, dataformats='HWC')

        outputs.yhat = torch.tensor([0])
        outputs.y = torch.tensor([0])

        accuracy = torch.tensor(sklearn.metrics.accuracy_score(y, yhat))
        f1 = torch.tensor(sklearn.metrics.f1_score(y, yhat, average='macro'))

        self.logger.experiment.add_scalar(f'{prefix}_accuracy', accuracy, self.current_epoch)
        self.logger.experiment.add_scalar(f'{prefix}_f1', f1, self.current_epoch)

        return outputs
