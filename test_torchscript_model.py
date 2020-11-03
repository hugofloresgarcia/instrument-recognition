import torch
import torchaudio
from instrument_recognition.utils import audio_utils
from instrument_recognition.data_modules.mdb import MDBDataset, MDBDataModule
from instrument_recognition.task import InstrumentDetectionTask, load_datamodule, load_model, get_task_parser

model = torch.jit.load('../torchscript_models/bigpapa-mdb-no1dbatchnorm.pt')
model.eval()

# parser = get_task_parser()

# hparams = parser.parse_args("--num_dataloader_workers 18 --batch_size 16 --accumulate_grad_batches 8 --learning_rate 0.0003 --log_gpu_memory True --log_epoch_metrics true --dataset medleydb --model tunedopenl3 --name bigpapa-mdb  --version 0 --gpuid 0".split())

# dm = load_datamodule(hparams)
# model = load_model(hparams, output_units=len(dm.dataset.classes))

# task = InstrumentDetectionTask.load_from_checkpoint('/home/hugo/lab/mono_music_sed/instrument_recognition/test-tubes/bigpapa-mdb/version_6/checkpoints/epoch=68-loss_val=0.12.ckpt', model, dm)

# dm = MDBDataModule(batch_size=1, sr=48000)
# dm.setup()
# for entry in dm.train_dataloader():
#     x = entry['X']
#     y = entry['y']

#     yhat = model(x)

#     yhat = torch.argmax(yhat, dim=1, keepdim=False)

#     print(y)
#     print(yhat)
#     print()


# dataset = MDBDataset()
# audio, sr = torchaudio.load('/home/hugo/data/philharmonia/all-samples/violin/violin_A4_15_fortissimo_arco-normal.mp3')
# audio = audio_utils.resample(audio.unsqueeze(0), sr, 48000)[:, :, 48000//2:int(48000*1.5)]
# # a = torch.randn((15, 1, 48000))
# # audio = torch.stack([*audio, *a])

# print(model(audio))
# print(dataset.classes[torch.argmax(model(audio), dim=1, keepdim=False)[0]])


# for i in range(20):
#     idx = torch.randint(0, len(dataset), (1,))
#     entry = dataset[idx]
#     audio = entry['X'].unsqueeze(0).cpu()
#     instrument = dataset.classes[entry['y']]

#     print(model(audio))
#     print(dataset.classes[torch.argmax(model(audio))])
#     print(instrument)
