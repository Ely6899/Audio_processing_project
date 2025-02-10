FREQUENCY_BIN_COUNT = 64
SAMPLE_RATE = 16000

N_FFT = 512
WINDOW_LENGTH = N_FFT
HOP_LENGTH = WINDOW_LENGTH // 2
MAX_SPECTOGRAM_DURATION_IN_SECONDS = 4.5

#Calculate the max number of samples for the target duration
MAX_SAMPLES = int(MAX_SPECTOGRAM_DURATION_IN_SECONDS * SAMPLE_RATE)

# Calculate the target number of frames for the spectrogram
#NOTE: For now, needs to be divisible by 8.
TARGET_FRAMES = (MAX_SAMPLES - WINDOW_LENGTH) // HOP_LENGTH + 1