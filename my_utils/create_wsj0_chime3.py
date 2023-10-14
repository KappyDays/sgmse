import os
from glob import glob
from librosa import load
from librosa.core import resample
import argparse
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from soundfile import write
from tqdm import tqdm
import pdb

# Python script for generating noisy mixtures for training
#
# Mix WSJ0 with CHiME3 noise with SNR sampled uniformly in [min_snr, max_snr]


min_snr = -5
max_snr = 5
sr = 16000

# def create_clean_noisy():
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("wsj0", type=str, help='path to WSJ0 directory')
    parser.add_argument("chime3", type=str,  help='path to CHiME3 directory')
    parser.add_argument("target", type=str, help='target path for training files')
    args = parser.parse_args()

    # Clean speech for training
    # file_path = "../../datasets/MyData/sgmse/train/"
    # print(args.wsj0 + '*.Clean.wav')
    speech_files = sorted(glob(args.wsj0 + '/*.wav', recursive=True))
    noise_files = glob(args.chime3 + '/*.wav', recursive=True)
    noise_files = [file for file in noise_files if (file[-7:-4] == "CH1")] # 1채널만 사용

    # Load CHiME3 noise files
    noises = []
    print('Loading CHiME3 noise files')
    for file in noise_files:
        noise = load(file, sr=None)[0]
        noises.append(noise)

    # Create target dir
    # clean_path = Path(os.path.join(args.target, 'clean'))
    noisy_path = Path(os.path.join(args.target, 'noisy'))
    # clean_path.mkdir(parents=True, exist_ok=True)
    noisy_path.mkdir(parents=True, exist_ok=True)

    # Initialize seed for reproducability
    np.random.seed(0)

    # Create files for training
    print('Create training files')
    print(len(speech_files))
    # for four in range(4):
    names = ['0000', '0001', '0002']
    # names = ['00dB']
    for name in names:
        for speech_file in tqdm(speech_files):
            s, _ = load(speech_file, sr=sr)

            snr_dB = np.random.uniform(min_snr, max_snr)
            # name = str(int(snr_dB))+'dB'
            noise_ind = np.random.randint(len(noises))
            speech_power = 1/len(s)*np.sum(s**2)

            # noise_file = speech_file[:-10] + ".Noise.wav"
            # n = load(noise_file, sr=None)[0]
            n = noises[noise_ind] # 노이즈 파일 불러오기
            
            start = np.random.randint(len(n)-len(s)) # noise file의 길이에서, speech file의 
            n = n[start:start+len(s)]
            noise_power = 1/len(n)*np.sum(n**2)
            noise_power_target = speech_power*np.power(10,-snr_dB/10)
            k = noise_power_target / noise_power
            n = n * np.sqrt(k)
            x = s + n

            file_name = speech_file.split('/')[-1]
            # write(os.path.join(clean_path, file_name), s, sr)
            write(os.path.join(noisy_path, file_name[:-4] + f'_{name}.wav'), x, sr, subtype="FLOAT")
        
        # if len(s) < len(n):
        #     # noise가 speech보다 길이가 긴 경우  noise를 랜덤 구간 추출한다.
        #     print(len(s), len(n))
        #     start = np.random.randint(len(n)-len(s)) # noise file의 길이에서, speech file의 
        #     n = n[start:start+len(s)]
        # elif len(s) > len(n):
        #     input()
        #     break
        # else:
        #     noise_power = 1/len(n)*np.sum(n**2)
        #     noise_power_target = speech_power*np.power(10,-snr_dB/10)
        #     k = noise_power_target / noise_power
        #     n = n * np.sqrt(k)
        #     x = s + n

        #     file_name = speech_file.split('/')[-1]
        #     write(os.path.join(clean_path, file_name), s, sr)
        #     write(os.path.join(noisy_path, file_name[:-9]+'Noisy.wav'), x, sr)

    # # Create files for validation
    # print('Create validation files')
    # for i, speech_file in enumerate(tqdm(valid_speech_files)):
    #     s, _ = load(speech_file, sr=sr)

    #     snr_dB = np.random.uniform(min_snr, max_snr)
    #     noise_ind = np.random.randint(len(noises))
    #     speech_power = 1/len(s)*np.sum(s**2)

    #     n = noises[noise_ind]
    #     start = np.random.randint(len(n)-len(s))
    #     n = n[start:start+len(s)]

    #     noise_power = 1/len(n)*np.sum(n**2)
    #     noise_power_target = speech_power*np.power(10,-snr_dB/10)
    #     k = noise_power_target / noise_power
    #     n = n * np.sqrt(k)
    #     x = s + n

    #     file_name = speech_file.split('/')[-1]
    #     write(os.path.join(valid_clean_path, file_name), s, sr)
    #     write(os.path.join(valid_noisy_path, file_name), x, sr)

    # # Create files for test
    # print('Create test files')
    # for i, speech_file in enumerate(tqdm(test_speech_files)):
    #     s, _ = load(speech_file, sr=sr)

    #     snr_dB = np.random.uniform(min_snr, max_snr)
    #     noise_ind = np.random.randint(len(noises))
    #     speech_power = 1/len(s)*np.sum(s**2)

    #     n = noises[noise_ind]
    #     start = np.random.randint(len(n)-len(s))
    #     n = n[start:start+len(s)]

    #     noise_power = 1/len(n)*np.sum(n**2)
    #     noise_power_target = speech_power*np.power(10,-snr_dB/10)
    #     k = noise_power_target / noise_power
    #     n = n * np.sqrt(k)
    #     x = s + n

    #     file_name = speech_file.split('/')[-1]
    #     write(os.path.join(test_clean_path, file_name), s, sr)
    #     write(os.path.join(test_noisy_path, file_name), x, sr)