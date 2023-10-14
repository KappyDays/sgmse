import os
import glob
import shutil
from tqdm import tqdm
import random

#####
# 0 ~ 20dB 모든 noisy 데이터를 trian, test로 나누고(9:1), 
# 각각의 폴더에 copy함으로써 저장
#####
def copy_noisy():
    count = 0
    file_list = []
    db_list = ['0dB', '5dB', '10dB', '15dB', '20dB']
    for db in db_list:
        file_list = glob.glob('../../dataset/army/' + db + '_noisy/*')
        filenumber = len(file_list)
        
        rate=0.1    #
        picknumber=int(filenumber*rate) #  rate     
        print(f"추출 개수: {picknumber}")         
        sample = random.sample(file_list, picknumber)  #    picknumber       
        # print (sample)
        for name in tqdm(sample):
            # shutil.move(fileDir+name, tarDir+name)
            shutil.copyfile(name, f'../../dataset/army/final/test/noisy/{name.split("/")[-1][:-4]}_{str(db)}.wav')
        
        
        # 데이터의 0.9는 train으로 사용
        temp1 = set(file_list)
        temp2 = set(sample)
        temp = list(temp1 - temp2)
        for name in tqdm(temp):
            shutil.copyfile(name, f'../../dataset/army/final/train/noisy/{name.split("/")[-1][:-4]}_{str(db)}.wav')
        
copy_noisy()
# 7218 - 72 = 7146
# 7146 ==> 9:1 = 6432:714