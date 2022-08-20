import shutil
import argparse
import torch.backends.cudnn as cudnn
import os
import scipy.io.wavfile as wavfile
import hifigan
import json
from matplotlib import pyplot as plt
import random
import yaml
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import time
from meldataset import build_dataloader
from tqdm import tqdm
from models import Generator, StyleEncoder



def plot_mel(data, titles):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]

    def add_axis(fig, old_ax):
        ax = fig.add_axes(old_ax.get_position(), anchor="W")
        ax.set_facecolor("None")
        return ax

    for i in range(len(data)):
        mel = data[i]
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

        ax1 = add_axis(fig, axes[i][0])
        ax1.set_xlim(0, mel.shape[1])
        ax1.set_ylabel("F0", color="tomato")
        ax1.tick_params(
            labelsize="x-small", colors="tomato", bottom=False, labelbottom=False
        )

    return fig


def get_hifigan(device):
    with open("hifigan/config.json", "r") as f:
        config = json.load(f)
    config = hifigan.AttrDict(config)
    vocoder = hifigan.Generator(config)
    ckpt = torch.load("hifigan/generator_universal.pth.tar")
    vocoder.load_state_dict(ckpt["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()
    vocoder.to(device)
    return vocoder


def load_checkpoint(checkpoint_directory, generator, style_encoder):
    print(checkpoint_directory)
    if os.path.exists(checkpoint_directory) is False:
        print("path does not exist!!!")
        return None
        
    gen_model = os.path.join(checkpoint_directory, checkpoint_directory.split("/")[-1] + ".pth") 
    
    params = torch.load(gen_model, map_location='cpu')

    if 'model_ema' not in params:
        params = params['model']
    else:
        params = params['model_ema']

    generator.load_state_dict(params["generator"])
    style_encoder.load_state_dict(params["style_encoder"])

    generator.eval()
    style_encoder.eval()

    style_encoder = style_encoder.to('cuda')
    generator = generator.to('cuda')
    print("Successfully loaded model !!!")

    return generator, style_encoder


def build_model(model_params={}):
    args = Munch(model_params)
    generator = Generator(args.dim_in, args.style_dim, args.max_conv_dim,
                          repeat_num=args.gen_n_repeat, res_num=args.gen_n_res,
                          multimlp=args.multimlp, in_style_dim=args.in_style_dim,
                          mel_max=args.mel_max, mel_min=args.mel_min)
    style_encoder = StyleEncoder(args.classes_num, stride=args.stride, norm=args.norm, pool=args.pool)

    return generator, style_encoder


def vc_generate(generator, style_encoder, source_data, target_data):

    with torch.no_grad():
        ref, ref_pred_label = style_encoder(target_data)
        mel = generator(source_data, ref)[0]
        return mel.squeeze(0)


def to_audio(mel, path, vocoder, mean, std):
    mel = (mel * std) + mean
    wavs = vocoder(mel)
    wavs = (wavs.view(-1).cpu().detach().numpy() * 32768.0).astype("int16")
    wavfile.write(path, 22050, wavs)

    fig = plot_mel(mel.cpu().numpy(), ["Synthetized Spectrogram"], )
    plt.savefig(path.replace(".wav", ".png"))
    plt.close()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='VCTK20', help='Path to the config file.')
    parser.add_argument('--output_path', type=str, default='.', help="outputs path")
    parser.add_argument('--display_num', type=int, default=8, help="Number of conversion samples")
    opts = parser.parse_args()

    config_path = './models/' + opts.config.split("/")[-1].split(".")[0] + '/config.yml'
    checkpoint_directory = './models/' + opts.config.split("/")[-1].split(".")[0]
    
    with open(config_path, 'r') as f:
        vc_model_config = yaml.safe_load(f)

    output_directory = os.path.join(opts.output_path + "/inferences", opts.config.split("/")[-1].split(".")[0])
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)


    generator, style_encoder = build_model(Munch(vc_model_config['model_params']))
    generator, style_encoder = load_checkpoint(checkpoint_directory, generator, style_encoder)
    setup_seed(1)

    vocoder = get_hifigan("cuda")
    vocoder.eval()

    val_path = vc_model_config.get('val_data', None)

    start = time.time()
    val_path = vc_model_config.get('val_data', None)
    batch_size = vc_model_config.get('batch_size', 10)
    device = vc_model_config.get('device')

    val_dataloader = build_dataloader(val_path, vc_model_config['mel_mean'], vc_model_config['mel_std'], batch_size=1, validation=False,
                                      num_workers=2, device=device)

    mean, std = vc_model_config.get('mel_mean'), vc_model_config.get('mel_std')
    
    countnum = 0
    for i, batch in enumerate(tqdm(val_dataloader)):

        batch = [b.to(device) for b in batch]
        source_data, source_label, target_data, _, target_label = batch

        source_label = source_label.cpu().numpy()[0]
        target_label = target_label.cpu().numpy()[0]

        result_source_audio_path = os.path.join(output_directory,
                                                "source_" + str(source_label) + "_" + str(countnum) + ".wav")
        result_target_audio_path = os.path.join(output_directory,
                                                "target_" + str(target_label) + "_" + str(countnum) + ".wav")
        result_conversion_audio_path_src_trg = os.path.join(output_directory, "conversion_" + str(source_label) +
                                                    "_to_" + str(target_label) + "_" + str(countnum) + ".wav")
        result_conversion_audio_path_trg_src = os.path.join(output_directory, "conversion_" + str(target_label) +
                                                    "_to_" + str(source_label) + "_" + str(countnum) + ".wav")
        countnum += 1

        convers_mel_src_trg = vc_generate(generator, style_encoder, source_data, target_data)
        convers_mel_trg_src = vc_generate(generator, style_encoder, target_data, source_data)
        
        to_audio(convers_mel_src_trg, result_conversion_audio_path_src_trg, vocoder, mean, std)
        to_audio(convers_mel_trg_src, result_conversion_audio_path_trg_src, vocoder, mean, std)
        to_audio(source_data.squeeze(0), result_source_audio_path, vocoder, mean, std)
        to_audio(target_data.squeeze(0), result_target_audio_path, vocoder, mean, std)

    print("generation finished !!!")
    end = time.time()
    print('total processing time: %.3f sec' % (end - start))

    