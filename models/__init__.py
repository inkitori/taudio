from .qwen2_5_omni_adapter import Qwen2_5OmniAdapter


def create_adapter(backend: str, model_id: str, load_in_8bit: bool, bidirectional_audio: bool):
    if backend.lower() in {"qwen2_5_omni", "qwen2.5_omni", "qwen2.5-omni", "qwen2_5-omni"}:
        return Qwen2_5OmniAdapter(model_id=model_id, load_in_8bit=load_in_8bit, bidirectional_audio=bidirectional_audio)

    raise ValueError(f"Unsupported backend '{
                     backend}'. Implement an adapter and register it here.")
