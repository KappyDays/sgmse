# Adapted from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/sampling.py
"""Various sampling methods."""
from scipy import integrate
import torch

from .predictors import Predictor, PredictorRegistry, ReverseDiffusionPredictor
from .correctors import Corrector, CorrectorRegistry

import pdb, sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from my_utils.exec_conf import inference_split_size, inference_block_num, split_size_1
from math import ceil
__all__ = [
    'PredictorRegistry', 'CorrectorRegistry', 'Predictor', 'Corrector',
    'get_sampler'
]


def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))


def get_pc_sampler(
    predictor_name, corrector_name, sde, score_fn, y,
    denoise=True, eps=3e-2, snr=0.1, corrector_steps=1, probability_flow: bool = False,
    intermediate=False,
    split=False,
    **kwargs
):
    """Create a Predictor-Corrector (PC) sampler.

    Args:
        predictor_name: The name of a registered `sampling.Predictor`.
        corrector_name: The name of a registered `sampling.Corrector`.
        sde: An `sdes.SDE` object representing the forward SDE.
        score_fn: A function (typically learned model) that predicts the score.
        y: A `torch.Tensor`, representing the (non-white-)noisy starting point(s) to condition the prior on.
        denoise: If `True`, add one-step denoising to the final samples.
        eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
        snr: The SNR to use for the corrector. 0.1 by default, and ignored for `NoneCorrector`.
        N: The number of reverse sampling steps. If `None`, uses the SDE's `N` property by default.

    Returns:
        A sampling function that returns samples and the number of function evaluations during sampling.
    """
    predictor_cls = PredictorRegistry.get_by_name(predictor_name)
    corrector_cls = CorrectorRegistry.get_by_name(corrector_name)
    score_fn.dnn.pc_trigger = True # PC_sampler에서는 deca_loss 반환 X
    predictor = predictor_cls(sde, score_fn, probability_flow=probability_flow)
    corrector = corrector_cls(sde, score_fn, snr=snr, n_steps=corrector_steps)
    # if split == True:
    #     print("#########나눠서 처리#######")

    def pc_sampler():
        split_size = inference_split_size
        block_size = split_size * inference_block_num # 256
        """The PC sampler function."""
        with torch.no_grad():
            if split == 'base':
                xt = sde.prior_sampling(y.shape, y).to(y.device)
                timesteps = torch.linspace(sde.T, eps, sde.N, device=y.device)
                for j in range(sde.N):
                    t = timesteps[j]
                    vec_t = torch.ones(y.shape[0], device=y.device) * t
                    xt, xt_mean = corrector.update_fn(xt, vec_t, y)
                    xt, xt_mean = predictor.update_fn(xt, vec_t, y)
                x_result = xt_mean if denoise else xt
                ns = sde.N * (corrector.n_steps + 1)
                
                score_fn.dnn.pc_trigger = False # pc_sampler 끝난 후 deca_loss 반환 o
                return x_result, ns
            ########################################################################
            elif split == 'original_split':
                temp_Y = []
                split_size = split_size_1
                if y.size(-1) // split_size > 0:
                    for i in range((y.size(-1) // split_size)):
                        temp_Y.append(y[..., i*split_size:(i+1)*split_size])
                    if y.size(-1) % split_size != 0:
                        temp_Y.append(y[..., (i+1)*split_size:y.size(-1)])
                else:
                    temp_Y.append(y)
                
                re = torch.Tensor().cuda()
                for temp_y in temp_Y:
                    # print(temp_y.shape)
                    # if temp_y.shape[-1] == 1152:
                        # pdb.set_trace()
                    xt = sde.prior_sampling(temp_y.shape, temp_y).to(temp_y.device)
                    timesteps = torch.linspace(sde.T, eps, sde.N, device=temp_y.device)
                    for i in range(sde.N):
                        t = timesteps[i]
                        vec_t = torch.ones(temp_y.shape[0], device=temp_y.device) * t
                        xt, xt_mean = corrector.update_fn(xt, vec_t, temp_y)
                        torch.cuda.empty_cache()
                        xt, xt_mean = predictor.update_fn(xt, vec_t, temp_y)
                    x_result = xt_mean if denoise else xt
                    ns = sde.N * (corrector.n_steps + 1)
                    re = torch.concat((re, x_result), dim=-1)
                
                score_fn.dnn.pc_trigger = False # pc_sampler 끝난 후 deca_loss 반환 o
                return re, ns                
            ########################### y 입력 나눠서 처리(GPU 메모리 문제) ##################
            elif split == 'overlab_split':
                temp_Y = []
                overlab_block_num = (y.size(-1)-block_size) // split_size + 1 # 3
                # pdb.set_trace()
                if overlab_block_num > 0:
                    for j in range(overlab_block_num):
                        temp_Y.append(y[..., j*(block_size-split_size):j*(block_size-split_size)+block_size])
                        # print("중간구간", j, j*(block_size-split_size), j*(block_size-split_size)+block_size)
                            
                    if (y.size(-1)-block_size) % split_size != 0: # 나머지 정보 있으면 추가
                        temp_Y.append(y[..., (j+1)*(block_size-split_size):y.size(-1)])
                        # print("마지막구간", j+1, (j+1)*(block_size-split_size), y.size(-1))
                else:
                    temp_Y.append(y)

                re = torch.Tensor().cuda()
                overlab_info = torch.zeros(0)
                # div_size = (split_size * (block_size-1)) // 2
                for iteration, temp_y in enumerate(temp_Y):
                    xt = sde.prior_sampling(temp_y.shape, temp_y).to(temp_y.device)
                    timesteps = torch.linspace(sde.T, eps, sde.N, device=temp_y.device)
                    for i in range(sde.N):
                        t = timesteps[i]
                        vec_t = torch.ones(temp_y.shape[0], device=temp_y.device) * t
                        xt, xt_mean = corrector.update_fn(xt, vec_t, temp_y)
                        xt, xt_mean = predictor.update_fn(xt, vec_t, temp_y)
                    x_result = xt_mean if denoise else xt
                    ns = sde.N * (corrector.n_steps + 1)
                    
                    # 조각에 overlab 및 hamming window 적용
                    # pdb.set_trace()
                    if temp_y.size(-1) == block_size:
                        # print("내부동작1", iteration, temp_y.shape)
                        signal_list = [] 
                        signal_list.append(x_result) # current_wav
                        # signal_list.append(x_result[..., split_size:]) # overlab_info2
                        # signal_list.append(x_result[..., block_size:block_size+split_size//2])
                        
                        ## istft를 통해 음성을 얻고 overlab함.
                        window = torch.hann_window(254, periodic=True)
                        sig_result = []
                        for signal in signal_list:
                            temp = score_fn.data_module.spec_back(signal)
                            sig_result.append(torch.istft(temp.squeeze(), n_fft=254, hop_length=128, 
                                                        window=window.to(temp.device), 
                                                        center=True))
                        
                        current_wav = sig_result[0]
                        div_size = current_wav.shape[0] // 2 + 128
                        # print("추가됩니다", overlab_info.shape[0], current_wav.shape)
                        current_wav[..., :overlab_info.shape[0]] += overlab_info.to(current_wav.device)
                        current_wav[..., :overlab_info.shape[0]] /= 2
                        
                        overlab_info = current_wav[..., div_size:]
                        
                        if iteration == len(temp_Y)-1: # 마지막 iter에서는 전부 넣음
                            re = torch.concat((re, current_wav))
                        else:
                            re = torch.concat((re, current_wav[..., :div_size]))
                    else:
                        # print("내부동작2", iteration, temp_y.shape)
                        current_wav = score_fn.data_module.spec_back(x_result)
                        current_wav = torch.istft(current_wav.squeeze(), n_fft=254, hop_length=128, 
                                                    window=torch.hann_window(254).to(current_wav.device), 
                                                    center=True)
                        if overlab_info.shape[0] >= current_wav.shape[0]:
                            current_wav += overlab_info[overlab_info.shape[0] - current_wav.shape[0]:]
                            current_wav /= 2
                        else:
                            current_wav[..., :overlab_info.shape[0]] += overlab_info.to(current_wav.device)
                            current_wav[..., :overlab_info.shape[0]] /= 2
                        
                        re = torch.concat((re, current_wav))
                        
                # print(re.shape)
                # pdb.set_trace()
                score_fn.dnn.pc_trigger = False # pc_sampler 끝난 후 deca_loss 반환 o
                return re, ns
            ########################################################################
            else:
                assert type(split) == str
    return pc_sampler


def get_ode_sampler(
    sde, score_fn, y, inverse_scaler=None,
    denoise=True, rtol=1e-5, atol=1e-5,
    method='RK45', eps=3e-2, device='cuda', **kwargs
):
    """Probability flow ODE sampler with the black-box ODE solver.

    Args:
        sde: An `sdes.SDE` object representing the forward SDE.
        score_fn: A function (typically learned model) that predicts the score.
        y: A `torch.Tensor`, representing the (non-white-)noisy starting point(s) to condition the prior on.
        inverse_scaler: The inverse data normalizer.
        denoise: If `True`, add one-step denoising to final samples.
        rtol: A `float` number. The relative tolerance level of the ODE solver.
        atol: A `float` number. The absolute tolerance level of the ODE solver.
        method: A `str`. The algorithm used for the black-box ODE solver.
            See the documentation of `scipy.integrate.solve_ivp`.
        eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
        device: PyTorch device.

    Returns:
        A sampling function that returns samples and the number of function evaluations during sampling.
    """
    predictor = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
    rsde = sde.reverse(score_fn, probability_flow=True)

    def denoise_update_fn(x):
        vec_eps = torch.ones(x.shape[0], device=x.device) * eps
        _, x = predictor.update_fn(x, vec_eps, y)
        return x

    def drift_fn(x, t, y):
        """Get the drift function of the reverse-time SDE."""
        return rsde.sde(x, t, y)[0]

    def ode_sampler(z=None, **kwargs):
        """The probability flow ODE sampler with black-box ODE solver.

        Args:
            model: A score model.
            z: If present, generate samples from latent code `z`.
        Returns:
            samples, number of function evaluations.
        """
        with torch.no_grad():
            # If not represent, sample the latent code from the prior distibution of the SDE.
            x = sde.prior_sampling(y.shape, y).to(device)

            def ode_func(t, x):
                x = from_flattened_numpy(x, y.shape).to(device).type(torch.complex64)
                vec_t = torch.ones(y.shape[0], device=x.device) * t
                drift = drift_fn(x, vec_t, y)
                return to_flattened_numpy(drift)

            # Black-box ODE solver for the probability flow ODE
            solution = integrate.solve_ivp(
                ode_func, (sde.T, eps), to_flattened_numpy(x),
                rtol=rtol, atol=atol, method=method, **kwargs
            )
            nfe = solution.nfev
            x = torch.tensor(solution.y[:, -1]).reshape(y.shape).to(device).type(torch.complex64)

            # Denoising is equivalent to running one predictor step without adding noise
            if denoise:
                x = denoise_update_fn(x)

            if inverse_scaler is not None:
                x = inverse_scaler(x)
            return x, nfe

    return ode_sampler
