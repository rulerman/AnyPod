from __future__ import annotations

from pathlib import Path
from types import MethodType
from typing import Any

import soundfile as sf
import torch
import torchaudio


def _load_audio_with_soundfile(audio_path: str | Path, target_sr: int) -> torch.Tensor:
    audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(audio).transpose(0, 1).contiguous()

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if int(sample_rate) != int(target_sr):
        wav = torchaudio.functional.resample(
            waveform=wav,
            orig_freq=int(sample_rate),
            new_freq=int(target_sr),
        )
    return wav


def patch_moss_processor_audio_loading(processor: Any) -> Any:
    if getattr(processor, "_anypod_soundfile_patch_applied", False):
        return processor

    if not hasattr(processor, "encode_audios_from_wav") or not hasattr(processor, "model_config"):
        return processor

    def _encode_audios_from_path_with_soundfile(
        self: Any,
        wav_path_list: str | list[str],
        n_vq: int | None = None,
    ):
        if isinstance(wav_path_list, str):
            wav_path_list = [wav_path_list]

        if len(wav_path_list) == 0:
            raise ValueError("Empty wav_path_list")

        target_sr = int(self.model_config.sampling_rate)
        wav_list = [
            _load_audio_with_soundfile(wav_path, target_sr)
            for wav_path in wav_path_list
        ]
        return self.encode_audios_from_wav(wav_list, target_sr, n_vq)

    processor.encode_audios_from_path = MethodType(_encode_audios_from_path_with_soundfile, processor)
    processor._anypod_soundfile_patch_applied = True
    return processor
