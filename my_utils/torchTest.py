import torch
import torchaudio

print(torch.cuda.is_available())

print(str(torchaudio.get_audio_backend()))