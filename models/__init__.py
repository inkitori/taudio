from .qwen2_5_omni_adapter import Qwen2_5OmniAdapter
# from .voxtral_adapter import VoxtralAdapter
# from .audio_flamingo3_adapter import AudioFlamingo3Adapter


def create_adapter(model_id: str, bidirectional_audio: bool, dtype: str, scaling_factor: int):
    if model_id.lower() in {"qwen/qwen2.5-omni-3b", "qwen/qwen2.5-omni-7b"}:
        return Qwen2_5OmniAdapter(model_id=model_id, bidirectional_audio=bidirectional_audio, dtype=dtype, scaling_factor=scaling_factor)
    # elif model_id.lower() in {"mistralai/voxtral-mini-3b-2507", "mistralai/voxtral-small-24b-2507"}:
    #     return VoxtralAdapter(model_id=model_id, bidirectional_audio=bidirectional_audio, dtype=dtype, scaling_factor=scaling_factor)
    # elif model_id.lower() in {"nvidia/audio-flamingo-3-hf"}:
    #     return AudioFlamingo3Adapter(model_id=model_id, bidirectional_audio=bidirectional_audio, dtype=dtype, scaling_factor=scaling_factor)
    else:
        raise ValueError(f"Unsupported model: {model_id}")
