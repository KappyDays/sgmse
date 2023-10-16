from os.path import join 
from glob import glob
from argparse import ArgumentParser
from soundfile import read
from tqdm import tqdm
from pesq import pesq
import pandas as pd
import pdb

from pystoi import stoi

from sgmse.util.other import energy_ratios, mean_std


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True, help='Directory containing the original test data (must have subdirectories clean/ and noisy/)')
    parser.add_argument("--enhanced_dir", type=str, required=True, help='Directory containing the enhanced data')
    args = parser.parse_args()

    test_dir = args.test_dir
    # clean_dir = join(test_dir, "clean/")
    clean_dir = '/mnt/ssd_mnt/kkr/sgmse/dataset/army/new_army/clean_test'
    # noisy_dir = join(test_dir, "test/noisy/")
    noisy_dir = join(test_dir, "noisy/")
    enhanced_dir = args.enhanced_dir

    data = {"filename": [], "pesq": [], "estoi": [], "si_sdr": [], "si_sir": [],  "si_sar": []}
    sr = 16000

    # pdb.set_trace()
    # Evaluate standard metrics
    noisy_files = sorted(glob('{}/*.wav'.format(noisy_dir)))
    for noisy_file in tqdm(noisy_files):
        ### 기본 filename
        # filename = '_'.join(noisy_file.split('/')[-1].split('_')[:-1]) + '.wav'
        ### SPINE noisy data는 wav 이름 맨 뒤에 _xxxx가 추가된 형태
        filename = '_'.join(noisy_file.split('/')[-1].split('_')[:-1]) + '.wav'
        temp = noisy_file.split('/')[-1]
        
        x, _ = read(join(clean_dir, filename))
        y, _ = read(noisy_file)
        n = y - x 
        # x_method, _ = read(join(enhanced_dir, filename))
        x_method, _ = read(join(enhanced_dir, temp))

        data["filename"].append(filename)
        data["pesq"].append(pesq(sr, x, x_method, 'wb'))
        data["estoi"].append(stoi(x, x_method, sr, extended=True))
        data["si_sdr"].append(energy_ratios(x_method, x, n)[0])
        data["si_sir"].append(energy_ratios(x_method, x, n)[1])
        data["si_sar"].append(energy_ratios(x_method, x, n)[2])

    # Save results as DataFrame    
    df = pd.DataFrame(data)

    # POLQA evaluation  -  requires POLQA license and server, uncomment at your own peril.
    # This is batch processed for speed reasons and thus runs outside the for loop.
    # if not basic:
    #     clean_files = sorted(glob('{}/*.wav'.format(clean_dir)))
    #     enhanced_files = sorted(glob('{}/*.wav'.format(enhanced_dir)))
    #     clean_audios = [read(clean_file)[0] for clean_file in clean_files]
    #     enhanced_audios = [read(enhanced_file)[0] for enhanced_file in enhanced_files]
    #     polqa_vals = polqa(clean_audios, enhanced_audios, 16000, save_to=None)
    #     polqa_vals = [val[1] for val in polqa_vals]
    #     # Add POLQA column to DataFrame
    #     df['polqa'] = polqa_vals

    # Print results
    print(enhanced_dir)
    #print("POLQA: {:.2f} ± {:.2f}".format(*mean_std(df["polqa"].to_numpy())))
    print("PESQ: {:.3f} ± {:.3f}".format(*mean_std(df["pesq"].to_numpy())))
    print("ESTOI: {:.3f} ± {:.3f}".format(*mean_std(df["estoi"].to_numpy())))
    print("SI-SDR: {:.3f} ± {:.3f}".format(*mean_std(df["si_sdr"].to_numpy())))
    print("SI-SIR: {:.3f} ± {:.3f}".format(*mean_std(df["si_sir"].to_numpy())))
    print("SI-SAR: {:.3f} ± {:.3f}".format(*mean_std(df["si_sar"].to_numpy())))

    # Save DataFrame as csv file
    df.to_csv(join(enhanced_dir, "_results.csv"), index=False)
    with open(join(enhanced_dir, "_results2.txt"), "w") as fp:
        fp.write("PESQ: {:.3f} ± {:.3f}\n".format(*mean_std(df["pesq"].to_numpy())))
        fp.write("ESTOI: {:.3f} ± {:.3f}\n".format(*mean_std(df["estoi"].to_numpy())))
        fp.write("SI-SDR: {:.3f} ± {:.3f}\n".format(*mean_std(df["si_sdr"].to_numpy())))
        fp.write("SI-SIR: {:.3f} ± {:.3f}\n".format(*mean_std(df["si_sir"].to_numpy())))
        fp.write("SI-SAR: {:.3f} ± {:.3f}".format(*mean_std(df["si_sar"].to_numpy())))        