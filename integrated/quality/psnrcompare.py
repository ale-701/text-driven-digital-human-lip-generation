import librosa
import numpy as np

def mel_cepstral_distortion(original, synthesized, sr=22050, n_mfcc=13):
    # Load audio files
    y1, sr1 = librosa.load(original, sr=sr)
    y2, sr2 = librosa.load(synthesized, sr=sr)

    # Ensure same length by padding shorter signal
    if len(y1) > len(y2):
        y2 = librosa.util.fix_length(y2, size=len(y1))
    else:
        y1 = librosa.util.fix_length(y1, size=len(y2))

    # Compute MFCCs
    mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=n_mfcc)
    mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2, n_mfcc=n_mfcc)

    # Compute the Euclidean distance between the MFCCs
    mcd = np.mean(np.sqrt(np.sum((mfcc1 - mfcc2) ** 2, axis=0)))

    return mcd

# Calculate MCD for the provided audio files
mcd_value = mel_cepstral_distortion('ref audio.wav', 'sound.wav')
print("MCD between 'ref audio.wav' and 'sound.wav':", mcd_value)
