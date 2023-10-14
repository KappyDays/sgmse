import torch
import time
from torchaudio import load

from pesq import pesq
from pystoi import stoi

import pdb, shutil

from .other import si_sdr, pad_spec

import sys, os
import torch.nn.functional as F
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from my_utils.exec_conf import load_clean_files, inference_split

# Settings
sr = 16000
snr = 0.5
N = 30
corrector_steps = 1
split = inference_split

def evaluate_model(model, num_eval_files, use_only_network):
    dataset_type = model.data_module.kwargs["dataset_type"]
    noisy_files = model.data_module.valid_set.noisy_files
    # pdb.set_trace()
    clean_files = load_clean_files(dataset_type, noisy_files, eval=True, model=model)
    
    # Select test files uniformly accros validation files
    total_num_files = len(noisy_files)
    indices = torch.linspace(0, total_num_files-1, num_eval_files, dtype=torch.int)
    clean_files = list(clean_files[i] for i in indices)
    noisy_files = list(noisy_files[i] for i in indices)


    # sep_time = int(sr*2.5)
    _pesq = 0
    _si_sdr = 0
    _estoi = 0
    # iterate over files
    with torch.no_grad():
        for (clean_file, noisy_file) in zip(clean_files, noisy_files):
            # Load wavs
            x, _ = load(clean_file)
            y, _ = load(noisy_file)
            # 사이즈가 큰 경우 줄임
            # pdb.set_trace()
            # if x.size(-1) > sep_time:
            #     x = x[:, :sep_time]
            #     y = y[:, :sep_time]
                
            # Normalize per utterance
            norm_factor = y.abs().max()
            y = y / norm_factor
            
            # Prepare DNN input
            Y = torch.unsqueeze(model._forward_transform(model._stft(y.cuda())), 0)
            Y = pad_spec(Y)
            # print(f'{noisy_file} {y.shape} {Y.shape}')
            
            y = y * norm_factor
            
            ## Reverse sampling
            T_orig = x.size(1)
            if use_only_network == "True": # DNN network만 사용, ncsnpp_deca에서 
                sample = model(Y)
                x_hat = model.to_audio(sample.squeeze(), T_orig)
            else: # Diffusion model 사용
                sampler = model.get_pc_sampler(
                    'reverse_diffusion', 'ald', Y.cuda(), N=N,  # N: number of reverse steps
                    corrector_steps=corrector_steps, snr=snr, split=split)
                sample, _ = sampler()
            
                if split == 'overlab_split':
                    x_hat = sample[:T_orig]
                else:
                    x_hat = model.to_audio(sample.squeeze(), T_orig)
                
            x_hat = x_hat * norm_factor        

            x_hat = x_hat.squeeze().cpu().numpy()
            x = x.squeeze().cpu().numpy()
            y = y.squeeze().cpu().numpy()
            
            _si_sdr += si_sdr(x, x_hat)
            try:
                _pesq += pesq(sr, x, x_hat, 'wb') 
            except:
                print(f'{noisy_file} pesq Error!!')
            _estoi += stoi(x, x_hat, sr, extended=True)
            
    return _pesq/num_eval_files, _si_sdr/num_eval_files, _estoi/num_eval_files