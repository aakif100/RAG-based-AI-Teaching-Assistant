# SPEECH TO TEXT USING FASTER WHISPER

from faster_whisper import WhisperModel

model = WhisperModel("medium", device="cuda", compute_type="float32")

segments, info = model.transcribe(
    "audios/12_Exercise 1 - Pure HTML Media Player.mp3",
    language="hi",
    task="translate"
)

print("Detected language:", info.language)
for segment in segments:  # or instead erite result[segments]
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

#----------------------------------------------------------------------
# second method of sample chunking  - writing smaple mp3 chunks to a json file
from faster_whisper import WhisperModel
import torch
import json

# Explicitly set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = WhisperModel("medium", device=device, compute_type="float32")
print(f"Model on: {device}")

segments, info = model.transcribe(
    "audios/sample.mp3",
    language="hi",
    task="translate"
)

# Convert segments to chunks format
chunks = []
for segment in segments:
    chunks.append({
        "start": segment.start,
        "end": segment.end,
        "text": segment.text
    })

print(chunks)

with open("sample_output.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=4)