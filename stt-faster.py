# SPEECH TO TEXT USING FASTER WHISPER

from faster_whisper import WhisperModel

model = WhisperModel("medium", device="cuda", compute_type="float32")

segments, info = model.transcribe(
    "audios/12_Exercise 1 - Pure HTML Media Player.mp3",
    language="hi",
    task="translate"
)

print("Detected language:", info.language)
for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")