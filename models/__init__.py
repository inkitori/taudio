from .qwen2_5_omni_adapter import Qwen2_5OmniAdapter


def create_adapter(model_id: str, bidirectional_audio: bool, dtype: str):
    if model_id.lower() in {"qwen/qwen2.5-omni-3b", "qwen/qwen2.5-omni-7b"}:
        return Qwen2_5OmniAdapter(model_id=model_id, bidirectional_audio=bidirectional_audio, dtype=dtype)
    else:
        raise ValueError(f"Unsupported model: {model_id}")
