import sounddevice as sd
from scipy.io.wavfile import write

fs = 16000        # Sample rate
seconds = 10      # Recording duration

print("Recording will start in 3 seconds...")
sd.sleep(3000)

print("Recording... Speak now!")

recording = sd.rec(
    int(seconds * fs),
    samplerate=fs,
    channels=1,
    dtype="int16"
)

sd.wait()

write("sample.wav", fs, recording)

print("Recording saved as sample.wav")