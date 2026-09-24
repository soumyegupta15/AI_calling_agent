import sounddevice as sd
import soundfile as sf
import numpy as np

print('Recording 3 seconds — speak now...')
audio = sd.rec(int(3 * 16000), samplerate=16000, channels=1, dtype='float32')
sd.wait()
sf.write('test_mic.wav', audio, 16000)
print('Saved test_mic.wav')
print('Max amplitude:', np.abs(audio).max())
