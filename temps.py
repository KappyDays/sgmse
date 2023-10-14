import random, shutil

# file_names = []
# with open('8s_files_from_valid.txt', 'r') as fp:
#     lines = fp.readlines()
#     for line in lines:
#         line = line.strip()
#         file_names.append(line.split(',')[0].split('/')[-1])

# my_list = random.sample(file_names, 327)
# # my_list = file_names

# for ff in my_list:
#     shutil.copy('/kkr/sgmse/dataset/wsj0_chime3/ori_valid/clean/'+ff, '/kkr/sgmse/dataset/wsj0_chime3/extrace_8s_from_valid/clean/')
#     shutil.copy('/kkr/sgmse/dataset/wsj0_chime3/ori_valid/noisy/'+ff, '/kkr/sgmse/dataset/wsj0_chime3/extrace_8s_from_valid/noisy/')
        
# import os
# os.listdir(/kkr/sgmse/dataset/wsj0_chime3/extrace_8s_test_from_valid/clean/)    

import os
file_list = os.listdir('/kkr/sgmse/dataset/wsj0_chime3/extrace_8s_from_valid/clean')
for file in file_list:
    file = '/kkr/sgmse/dataset/wsj0_chime3/extrace_8s_from_valid/clean/' + file
    
    shutil.copy(file, '/kkr/sgmse/dataset/wsj0_chime3/test/clean')