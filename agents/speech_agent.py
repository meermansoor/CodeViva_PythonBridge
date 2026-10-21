import os

from faster_whisper import WhisperModel
import librosa
import numpy as np

# Load local Faster-Whisper model ('base' for fast local CPU inference)
print("Loading Faster-Whisper model...")
model = WhisperModel("base", device="cpu", compute_type="int8")

def check_audio_authenticity(audio_path: str) -> dict:
    """
    Analyzes basic acoustic features (zero-crossing rate & spectral rollover)
    to flag unnatural synthetic/TTS voice patterns.
    """
    try:
        y, sr = librosa.load(audio_path, sr=None)
        
        # Spectral centroids / Zero crossing rate to check audio variation
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        
        # Simple heuristic check: synthetic audio often has overly flat or uniform spectral signatures
        is_authentic = True
        reason = "Human voice pitch & frequency variation detected."

        if spectral_flatness < 0.0001 or zcr == 0:
            is_authentic = False
            reason = "Suspicious audio signature (Possible static synthetic TTS)."

        return {
            "is_authentic": is_authentic,
            "reason": reason,
            "flatness": float(spectral_flatness)
        }
    except Exception as e:
        return {"is_authentic": True, "reason": f"Analysis skipped: {str(e)}"}

def transcribe_and_verify(audio_path: str) -> dict:
    """
    Main Agent 1 function: Authenticates audio signal and transcribes it to text.
    """
    if not os.path.exists(audio_path):
        return {"error": "Audio file not found"}

    # 1. Authenticity check
    auth_result = check_audio_authenticity(audio_path)
    
    # 2. Transcription via Faster-Whisper with VAD filter to skip silence
    segments, info = model.transcribe(audio_path, beam_size=5, vad_filter=True)
    
    transcription_text = " ".join([segment.text for segment in segments]).strip()
    
    return {
        "text": transcription_text,
        "is_authentic": auth_result["is_authentic"],
        "auth_notes": auth_result["reason"]
    }

# --- Quick Code Test ---
if __name__ == "__main__":
    print("Agent 1 (Speech Authenticity & Transcription) is ready!")