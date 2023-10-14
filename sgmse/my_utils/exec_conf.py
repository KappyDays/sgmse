from glob import glob
from os.path import join
import pdb

# backbones = ["NCSNpp", "Multiple_NCSNpp"]
# choose_backbone = backbones[1]
# TwoCross = True

## sampling -> _init_.py (inference 시 y의 입력 을 나눠서 처리, GPU 메모리 때문)

split_type = ['base', 'original_split', 'overlab_split']
inference_split_size, inference_block_num = 256, 4
inference_split = split_type[0]
split_size_1 = 1152

# noisy 파일을 기준으로 clean file list를 생성
def load_clean_files(d_type, noisy_files=False, data_dir=False, subset=False, eval=False, model=False):
    #하나의 clean folder를 여러번 사용 (중복된 이름의 noisy가 여러 개인 경우)
    if d_type == "default":
        if eval == True:
            clean_files = model.data_module.valid_set.clean_files
        else:
            clean_files = sorted(glob(join(data_dir, subset) + '/clean/*.wav'))
    if d_type == "use_clean_multiple":
        clean_data_dir = "dataset/army/new_army/clean/"
        # if eval == True:
        #     clean_files = []
        #     for noisy_file in noisy_files:
        #         clean_files.append(f'{clean_data_dir}{"_".join(noisy_file.split("/")[-1].split("_")[:-1])}.wav')
        # else:
        clean_files = []
        for noisy_file in noisy_files:
            clean_files.append(clean_data_dir + '_'.join(noisy_file.split('/')[-1].split('_')[:-1]) + ".wav")
    # 다른 dir의 clean data 사용할 경우
    elif d_type == "use_other_clean_dir":
        if eval == True:
            # To Do: inference.py에서 사용되는 clean loader 작성
            ## Ori Code
            # clean_files = []
            # # clean 경로 입력
            # other_dir_path = "dataset/army/military_noise"
            # for file in noisy_files:
            #     clean_files.append(f'{other_dir_path}/clean/{"_".join(file.split("/")[-1].split("_"))}')            
            pass
        else:
            # To Do: use_clean_multiple 코드 그대로 사용
            pass
            ## Ori Code
            # # clean 경로 입력
            # other_dir_path = "dataset/army/military_noise"
            # clean_files = other_dir_path + '/clean/' + '_'.join(noisy_files[i].split('/')[-1].split('_'))

    return clean_files