import glob, shutil, os

# subset = ['train/noisy/', 'valid/noisy/', 'test/noisy/']
# dBs = ['0dB', '5dB', '10dB', '15dB','20dB']
target_file_path = '/mnt/aiter/kkr/sgmse/dataset/army_multiple_UNet_noisy_noise'
target_file_list = os.listdir(f'{target_file_path}/train/noisy')

# no utterance 저장할 폴더
no_utter_folder = 'no_utter_file'
save_path = f'{target_file_path}/{no_utter_folder}'
os.mkdir(save_path)

remove_count = 0
with open('no_utterances_data_list.txt', 'r') as fp:
    files = fp.readlines()
    
    for file in files:
        file = file.split('/')[-1].strip()
        
        for target_file in target_file_list:
            if file == target_file:
                remove_count += 1
                print(file)
                
                with open(f'{save_path}/removed_data.txt' ,'a') as fr:
                    shutil.move(f'{target_file_path}/train/noisy/{file}', save_path)
                    fr.write(f'{file}\n')
print(f'제거된 개수: {remove_count}')                    