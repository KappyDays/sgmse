import os, shutil
import pdb

path = '/mnt/aiter/kkr/sgmse/dataset/army_multiple_UNet_speech_noise/'
files = os.listdir(path + 'valid/noisy')

valid_list = []
with open('valid_data_list.txt', 'r') as fp:
    lines = fp.readlines()
    for line in lines:
        # print(line)
        line = '_'.join(line.strip().split('/')[-1].split('_')[:-1])+'.wav'
        # print(line)
        valid_list.append(line)

# print(all_list)
for file in valid_list:
    # pdb.set_trace()
    # file_name = '_'.join(file.split('_')[:-1]) + '.wav'
    # print(file_name)
    shutil.move(f'{path}/train/noisy/{file}', f'{path}/used_for_valid_NotTrain/{file}')
    