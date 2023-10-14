# [이미지 파일의 이름과 같은 txt파일 옮기기]

import os, shutil
from tqdm import tqdm
import pdb

# 이미지 폴더
img_path = "../../datasets/MyData/sgmse/test/"
# txt 폴더
txt_path = "../../datasets/isolated_ext/dt05_bus_simu/"
# 옮길 목적지
target_path = "../../datasets/MyData/sgmse/test/" 



img_list = os.listdir(img_path)
txt_list = os.listdir(txt_path)

print("이미지 파일 개수: ", len(img_list))
print("txt 파일 개수: ", len(txt_list))
count = 0
#shutil.move(fileDir+name, tarDir+name)
my_file_list = []
for txt in txt_list:
    if txt.split('.')[2] == 'Clean':
        my_file_list.append(txt)    

print("내 파일 개수: ", len(my_file_list))
for file in tqdm(my_file_list):
    splited = file.split('.')
    txt = splited[0] + '.' + splited[1]
    for img in img_list:
        splited = img.split('.')
        img = splited[0] + '.' + splited[1]        
        if img == txt:
            # shutil.move(txt_path + txt + ".txt", target_path + txt + ".txt")
            shutil.copy(txt_path + txt + ".Clean.wav", target_path + txt + ".Clean.wav")
            # pdb.set_trace()
            count += 1
            img_list.remove(img + ".Noise.wav")
            break
        
print("옮기기 성공한 개수: ", count)