import numpy as np

def clamp(n, smallest, largest): return max(smallest, min(n, largest))

# Pad 16khz raw audio frames, where the encoded audio is 40ms per embedding
def pad_audio(audio_frames, padding):
	pad_length = 16 * 40 * padding
	audio_frames = np.pad(audio_frames, (0, pad_length), mode='constant')

	return audio_frames