from soundfile import read
import os, glob
from tqdm import tqdm

folder_path = "datasets/KAIST/0dB/clean/"
clean_file_list = glob.glob("../../" + folder_path + "*")

for file_to_read in tqdm(clean_file_list):
    wav_file, sr = read(file_to_read)

    if sr != 16000:
        print(f'sr이 다름 {file_to_read} {sr}')
        with open('different_sr.txt', 'a') as fp:
            fp.write(f'{file_to_read},{str(sr)}\n')
            
    speech_len = len(wav_file) / sr
    if speech_len < 1:
        with open('short_speech_1.txt', 'a') as fp:
            fp.write(f'{file_to_read},{speech_len},{len(wav_file)},{str(sr)}\n')
            