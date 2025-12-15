from .qwen2_5_omni_adapter import Qwen2_5OmniAdapter


def create_adapter(model_id: str, bidirectional_audio: bool, dtype: str, scaling_factor: int):
    if model_id.lower() in {"qwen/qwen2.5-omni-3b", "qwen/qwen2.5-omni-7b"}:
        return Qwen2_5OmniAdapter(model_id=model_id, bidirectional_audio=bidirectional_audio, dtype=dtype, scaling_factor=scaling_factor)
    elif model_id.lower() in {"mistralai/voxtral-mini-3b-2507", "mistralai/voxtral-small-24b-2507"}:
        try:
            from .voxtral_adapter import VoxtralAdapter
            return VoxtralAdapter(model_id=model_id, bidirectional_audio=bidirectional_audio, dtype=dtype, scaling_factor=scaling_factor)
        except ModuleNotFoundError:
            raise ModuleNotFoundError(f"Can't import Voxtral (might need to update transformers)")
    elif model_id.lower() in {"nvidia/audio-flamingo-3-hf"}:
        try:
            from .audio_flamingo3_adapter import AudioFlamingo3Adapter
            return AudioFlamingo3Adapter(model_id=model_id, bidirectional_audio=bidirectional_audio, dtype=dtype, scaling_factor=scaling_factor)
        except ModuleNotFoundError:
            raise ModuleNotFoundError(f"Can't import Audio Flamingo (might need to update transformers)")
    else:
        raise ValueError(f"Unsupported model: {model_id}")
