# [이미지 파일의 이름과 같지 않은 txt파일 옮기기]

import os, shutil
from tqdm import tqdm
import pdb

# 이미지 폴더
img_path = "../../datasets/MyData/sgmse/train/"
# txt 폴더
txt_path = "../../datasets/MyData/sgmse/valid/" #isolated_ext/dt05_str_simu/"
# 옮길 목적지
target_path = "../../datasets/MyData/sgmse/test/"



img_list = os.listdir(img_path)
txt_list = os.listdir(txt_path)

print("이미지 파일 개수: ", len(img_list))
print("txt 파일 개수: ", len(txt_list))
#shutil.move(fileDir+name, tarDir+name)
my_img_list = []
for img in img_list:
    if img.split('.')[2] == 'Clean':
        my_img_list.append(img)
        
my_file_list = []
for txt in txt_list:
    if txt.split('.')[2] == 'Noise':
        my_file_list.append(txt)    

count = 0
my_img_list = img_list.copy()
my_file_list = txt_list.copy()
print("(검사 대상)내 파일 개수: ", len(my_img_list))
print("(검사 대상)내 파일 개수: ", len(my_file_list))
for img in tqdm(my_img_list):
    splited = img.split('.')
    img = splited[0] + '.' + splited[1] + '.' + splited[2]
    for file in my_file_list:
        splited = file.split('.')
        txt = splited[0] + '.' + splited[1] + '.' + splited[2]
        if img == txt:
            print("??")
            # input()
            pdb.set_trace()
            # os.remove(target_path + txt + ".Noise.wav")
            # os.remove(target_path + txt + ".Clean.wav")
            # shutil.move(txt_path + txt + ".txt", target_path + txt + ".txt")
            # shutil.copy(txt_path + txt + ".Noise.wav", target_path + txt + ".Noise.wav")
            # pdb.set_trace()
            # count += 1
            # my_file_list.remove(img + ".Noise.wav")
            # my_file_list.remove(img + ".Clean.wav")
            break
        
print("지우기 성공한 개수: ", count)