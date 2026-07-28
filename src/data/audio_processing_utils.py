import torchaudio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as aF
import soundfile as sf
from julius import ResampleFrac
import random
from scipy.io import wavfile
import numpy as np
import pyloudnorm as pyln
from functools import lru_cache
from types import SimpleNamespace
import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*torchaudio\._backend\.utils\.info has been deprecated.*",
    category=UserWarning,
)
@lru_cache(maxsize=32)
def get_resampler(src_sr, tar_sr):
    return torchaudio.transforms.Resample(orig_freq=src_sr, new_freq=tar_sr)

MAX_INT16 = 32768.0


def get_audio_info(filepath):
    """Return audio metadata with torchaudio first, then soundfile fallback."""
    try:
        return torchaudio.info(filepath)
    except Exception:
        sf_meta = sf.info(filepath)
        return SimpleNamespace(
            sample_rate=int(sf_meta.samplerate),
            num_frames=int(sf_meta.frames),
            num_channels=int(sf_meta.channels),
        )


def _load_audio_with_soundfile(filepath, start=None, end=None):
    start_frame = int(start) if start is not None else 0
    stop_frame = int(end) if end is not None else None
    samples, _ = sf.read(
        filepath,
        start=start_frame,
        stop=stop_frame,
        dtype="float32",
        always_2d=True,
    )
    return torch.from_numpy(samples.T)

def load_audio(filepath, start=None, end=None, load_mode='torchaudio'):

    if load_mode == 'torchaudio':
        num_frames = end - start if (end is not None and start is not None) else None
        try:
            waveform, _ = torchaudio.load(
                filepath,
                frame_offset=start or 0,
                num_frames=num_frames,
            )
        except Exception:
            waveform = _load_audio_with_soundfile(filepath, start=start, end=end)
    elif load_mode == 'scipy':
        # make use of mmap to access segment from large audio files
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html
        _, waveform = wavfile.read(filepath, mmap=True)
        waveform = torch.from_numpy(waveform[start:end]/MAX_INT16).float().unsqueeze(0)
    elif load_mode == 'soundfile':
        waveform = _load_audio_with_soundfile(filepath, start=start, end=end)

    return waveform

def load_waveform(filepath,
                  start=None,
                  tar_sr=None,
                  tar_len=None,
                  load_mode: str = 'torchaudio',
                  return_start=False,
                  known_sr=None,
                  known_frames=None):
    """
    Args:
        filepath (str): filepath to the audio file
        tar_sr (float): target sampling rate
        tar_len (float): target length in seconds
    Returns:
        torch tensor: 1D waveform
    """
    if known_sr is not None and known_frames is not None:
        src_sample_rate = known_sr
        src_len = known_frames
    else:
        audio_metadata = get_audio_info(filepath)
        src_len = audio_metadata.num_frames
        src_sample_rate = audio_metadata.sample_rate

    if tar_len is not None:
        tar_len = int(tar_len * src_sample_rate)
        if start is None:
            start_frame = random.randint(0, src_len - tar_len) if src_len > tar_len else 0
            start = start_frame/src_sample_rate
        else:
            start_frame = int(start * src_sample_rate)
            if start_frame > src_len:
                print('start time exceeds audio length', filepath)
                # exit()
                # start_frame = 0
                out_wav = torch.zeros(int(tar_len / src_sample_rate * tar_sr)) if tar_sr else torch.zeros(tar_len)
                if return_start:
                    return out_wav, start
                else:
                    return out_wav
        waveform = load_audio(filepath, start=start_frame, end=start_frame+tar_len, load_mode=load_mode)

        cur_len = waveform.shape[-1]
        if cur_len < tar_len:
            waveform = F.pad(waveform, (0, tar_len-cur_len), 'constant', 0)
    else:
        waveform = load_audio(filepath, load_mode=load_mode)
        start = 0.0

    if tar_sr is not None and src_sample_rate != tar_sr:
        resampler = get_resampler(src_sample_rate, tar_sr)
        waveform = resampler(waveform)
    # convert to mono
    waveform = waveform.mean(dim=0)
    if return_start:
        return waveform, start
    else:
        return waveform

def add_reverb_noise(audio, reverb=None, noise=None, snr_db=0, target_len=1):
    """
    Add noise and reverberation

    Args:
        audio (_type_): _description_
        reverb (_type_): _description_
        noise (_type_): _description_
        snr_db (_type_): _description_
        target_len (_type_): _description_

    Returns:
        _type_: _description_
    """

    noisy_speech = aF.add_noise(audio.unsqueeze(0), noise.unsqueeze(0), snr_db).squeeze(0)
    if reverb is not None:
        reverb = reverb / torch.linalg.vector_norm(reverb, ord=2)
        reverb = reverb / reverb.abs().max()
        noisy_speech = aF.fftconvolve(noisy_speech, reverb)

    # noisy_speech = noisy_speech / torch.linalg.vector_norm(noisy_speech, ord=2) * torch.linalg.vector_norm(noisy_speech, ord=2)

    if len(noisy_speech) > target_len:
        noisy_speech = noisy_speech[:target_len]

    return noisy_speech


class HighPass(nn.Module):
    def __init__(self,
                 nfft=1024,
                 hop=256,
                 ratio=(1 / 6, 1 / 3, 1 / 2, 2 / 3, 3 / 4, 4 / 5, 5 / 6,
                        1 / 1)):
        super().__init__()
        self.nfft = nfft
        self.hop = hop
        self.register_buffer('window', torch.hann_window(nfft), False)
        f = torch.ones((len(ratio), nfft//2 + 1), dtype=torch.float)
        for i, r in enumerate(ratio):
            f[i, :int((nfft//2+1) * r)] = 0.
        self.register_buffer('filters', f, False)

    #x: [B,T], r: [B], int
    @torch.no_grad()
    def forward(self, x, r):
        if x.dim()==1:
            x = x.unsqueeze(0)
        T = x.shape[1]
        x = F.pad(x, (0, self.nfft), 'constant', 0)
        stft = torch.stft(x,
                          self.nfft,
                          self.hop,
                          window=self.window,
                          )#return_complex=False)  #[B, F, TT,2]
        stft *= self.filters[r].view(*stft.shape[0:2],1,1 )
        x = torch.istft(stft,
                        self.nfft,
                        self.hop,
                        window=self.window,
                        )#return_complex=False)
        x = x[:, :T].detach()
        return x


class LowPass(nn.Module):
    def __init__(self,
                 nfft=1024,
                 hop=256,
                 ratio=(1/6, 1/3, 1/2, 2/3, 3/4, 4/5, 5/6, 1/1)):
        super().__init__()
        self.nfft = nfft
        self.hop = hop
        self.register_buffer('window', torch.hann_window(nfft), False)
        f = torch.ones((len(ratio), nfft//2 + 1), dtype=torch.float)
        for i, r in enumerate(ratio):
            f[i, int((nfft//2+1) * r):] = 0.
        self.register_buffer('filters', f, False)

    #x: [B,T], r: [B], int
    @torch.no_grad()
    def forward(self, x, r):
        if x.dim()==1:
            x = x.unsqueeze(0)
        T = x.shape[1]
        x = F.pad(x, (0, self.nfft), 'constant', 0)
        stft = torch.stft(x,
                          self.nfft,
                          self.hop,
                          window=self.window,
                          return_complex=True)  #[B, F, TT,2]
        stft *= self.filters[r].view(*stft.shape[0:2],1 )
        x = torch.istft(stft,
                        self.nfft,
                        self.hop,
                        window=self.window,
                        return_complex=False)
        x = x[:, :T].detach()
        return x

class SegmentMixer(nn.Module):

    """
    https://github.com/Audio-AGI/AudioSep/blob/main/data/waveform_mixers.py


    """
    def __init__(self, max_mix_num, lower_db, higher_db):
        super(SegmentMixer, self).__init__()

        self.max_mix_num = max_mix_num
        self.loudness_param = {
            'lower_db': lower_db,
            'higher_db': higher_db,
        }

    def __call__(self, waveforms, noise_waveforms):

        batch_size = waveforms.shape[0]
        noise_indices = torch.randperm(batch_size)

        data_dict = {
            'segment': [],
            'mixture': [],
        }

        for n in range(batch_size):

            segment = waveforms[n].clone()

            # random sample from noise waveforms
            noise = noise_waveforms[noise_indices[n]]
            noise = dynamic_loudnorm(audio=noise, reference=segment, **self.loudness_param)

            mix_num = random.randint(2, self.max_mix_num)
            assert mix_num >= 2

            for i in range(1, mix_num):
                next_segment = waveforms[(n + i) % batch_size]
                rescaled_next_segment = dynamic_loudnorm(audio=next_segment, reference=segment, **self.loudness_param)
                noise += rescaled_next_segment

            # randomly normalize background noise
            noise = dynamic_loudnorm(audio=noise, reference=segment, **self.loudness_param)

            # create audio mixyure
            mixture = segment + noise

            # declipping if need be
            max_value = torch.max(torch.abs(mixture))
            if max_value > 1:
                segment *= 0.9 / max_value
                mixture *= 0.9 / max_value

            data_dict['segment'].append(segment)
            data_dict['mixture'].append(mixture)

        for key in data_dict.keys():
            data_dict[key] = torch.stack(data_dict[key], dim=0)

        # return data_dict
        return data_dict['segment'], data_dict['mixture']


def rescale_to_match_energy(segment1, segment2):

    ratio = get_energy_ratio(segment1, segment2)
    rescaled_segment1 = segment1 / ratio
    return rescaled_segment1


def get_energy(x):
    return torch.mean(x ** 2)


def get_energy_ratio(segment1, segment2):

    energy1 = get_energy(segment1)
    energy2 = max(get_energy(segment2), 1e-10)
    ratio = (energy1 / energy2) ** 0.5
    ratio = torch.clamp(ratio, 0.02, 50)
    return ratio


def dynamic_loudnorm(audio, reference, lower_db=-10, higher_db=10):
    rescaled_audio = rescale_to_match_energy(audio, reference)
    delta_loudness = random.randint(lower_db, higher_db)
    gain = np.power(10.0, delta_loudness / 20.0)

    return gain * rescaled_audio

def normalize_loudness(audio, sr=32000, target_lufs=-23.0):
    """
    Normalize audio to target LUFS.
    Args:
        audio: torch Tensor [1, T] or [T]
        sr: sampling rate
        target_lufs: target loudness in LUFS
    """
    device = audio.device
    # ensure input is [1, T] for consistency in return
    if audio.dim() == 1:
        audio_in = audio.unsqueeze(0)
    else:
        audio_in = audio

    # pyloudnorm expects [samples, channels] or [samples] numpy array
    # We use [1, T], so we convert to [T] for mono
    audio_np = audio_in.squeeze(0).detach().cpu().numpy()

    # Safety check for silence or near silence
    if np.max(np.abs(audio_np)) < 1e-9:
        return audio_in

    meter = pyln.Meter(sr)
    try:
        loudness = meter.integrated_loudness(audio_np)
        # check if loudness is finite
        if np.isinf(loudness) or np.isnan(loudness):
            return audio_in

        normalized_audio_np = pyln.normalize.loudness(audio_np, loudness, target_lufs)
        normalized_audio = torch.from_numpy(normalized_audio_np).unsqueeze(0)
        return normalized_audio.to(device)
    except Exception as e:
        # Fallback if normalization fails (e.g. signal too short)
        return audio_in

def normalize_rms(audio, target_rms=0.1):
    """
    Normalize audio to target RMS.
    Args:
        audio: torch Tensor [1, T] or [T]
        target_rms: target RMS level
    """
    device = audio.device
    if audio.dim() == 1:
        audio_in = audio.unsqueeze(0)
    else:
        audio_in = audio

    # [1, T] -> [T]
    audio_np = audio_in.squeeze(0).detach().cpu().numpy()

    rms = np.sqrt(np.mean(audio_np**2))
    if rms < 1e-9:
        return audio_in

    scalar = target_rms / rms

    # Scale tensor
    normalized_audio = audio_in * scalar
    return normalized_audio

# decayed
def random_loudness_norm(audio, lower_db=-35, higher_db=-15, sr=32000):
    device = audio.device
    audio = audio.squeeze(0).detach().cpu().numpy()
    # randomly select a norm volume
    norm_vol = random.randint(lower_db, higher_db)

    # measure the loudness first
    meter = pyln.Meter(sr) # create BS.1770 meter
    loudness = meter.integrated_loudness(audio)
    # loudness normalize audio
    normalized_audio = pyln.normalize.loudness(audio, loudness, norm_vol)

    normalized_audio = torch.from_numpy(normalized_audio).unsqueeze(0)

    return normalized_audio.to(device)
