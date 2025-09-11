ASSISTANT_ID = 77091
BEGIN_AUDIO_ID = 151647
END_AUDIO_ID = 151648
# 40 milliseconds per embedding (from technical report) THIS ONLY APPLIES TO QWEN2.5 OMNI
# For example, 10 seconds of audio would be 10 * 1000 = 10000 milliseconds, which would be 10000 / 40 = 250 embeddings.
SECONDS_TO_EMBEDDING = (1000) * (1 / 40)