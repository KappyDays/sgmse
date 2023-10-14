from pesq import pesq
import glob
from torchaudio import load
from tqdm import tqdm


file_list = glob.glob('../../dataset/army/final/clean/*')
# print(file_list)

result = 0
count = 0
for file in tqdm(file_list):
    x, _ = load(file)
    x = x.squeeze().cpu().numpy()
    try:
        result += pesq(16000, x, x, 'wb')
        count += 1
    except:
        with open('no_utter.txt', 'a') as fp:
            fp.write(f'{file}\n')
            
print(result / count)