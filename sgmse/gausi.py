import numpy as np
import pdb

def generate_gaussian_noise(mu, sigma, size):
    noise = np.random.normal(mu, sigma, size)
    return noise

# 평균과 표준편차를 설정합니다
mu = 0     # 평균
sigma = 1  # 표준편차
size = 10  # 샘플 수

# 가우시안 노이즈 샘플을 생성합니다
noise_samples = generate_gaussian_noise(mu, sigma, size)

print("Generated Gaussian Noise Samples:", noise_samples)
pdb.set_trace()
