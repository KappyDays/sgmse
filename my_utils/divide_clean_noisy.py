import os, random, shutil
from tqdm import tqdm

def get_file_num(file_path):
    file_list = os.listdir(file_path)
    print(f"{file_path}의 파일 개수는 {len(file_list)}")
    return file_list

# dt05_bus = get_file_num("MyData/sgmse/train")

def moveFile(fileDir, tarDir):
        all_file_list = os.listdir(fileDir)
        # my_file_list = []
        # for file in all_file_list:
        #     if file.split('.')[2] == 'Noise':
        #         my_file_list.append(file)
        # print(len(my_file_list))
        # filenumber = len(all_file_list)
        # rate=0.1    #
        # picknumber=int(filenumber*rate) #  rate
        picknumber = 3570
        print(f"추출 개수: {picknumber}")         
        sample = random.sample(all_file_list, picknumber)  #    picknumber       
        # print (sample)
        for name in tqdm(sample):
            shutil.move(fileDir+name, tarDir+name)
            # shutil.copy(fileDir+name, tarDir+name)
        return

if __name__ == '__main__':
    fileDir = "../dataset/army/non_military_noise/train/noisy/"
    tarDir = "../dataset/army/non_military_noise/test/noisy/"
    moveFile(fileDir, tarDir)
# print(len(dt05_bus))
# print(type(dt05_bus))
# /mnt/aiter/kkr/datasets/isolated_ext/dt05_bus_simu