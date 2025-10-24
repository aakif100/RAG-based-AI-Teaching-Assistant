# stt means SPEECH-TO-TEXT

# ---------------------------------------------------------------------
# cwh version - doesnt explicitly check for cuda (gpu)

# import whisper
# import torch

# model = whisper.load_model("tiny")
# result = model.transcribe(audio ="audios/12_Exercise 1 - Pure HTML Media Player.mp3" ,
#                           language="hi",
#                           task="translate")

# print(result["text"])

# ---------------------------------------------------------------------


#  c version explicitly checks for cuda (gpu) – alternate code

# import whisper
# import torch

# #  Explicitly set device
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")

# model = whisper.load_model("medium", device=device)  #  Pass device here and it uses gpu if available
# print(f"Model on: {next(model.parameters()).device}") # not really necessary, just to confirm device

# result = model.transcribe(
#     audio="audios/12_Exercise 1 - Pure HTML Media Player.mp3",
#     language="hi",
#     task="translate",
#     fp16 = False
# )

# print(result["text"])





# ------------------------------------------------------------------------
# second method of sample chunking  - writing smaple mp3 chunks to a json file

# here done to see speech to text chunking and writing to a json file with sample mp3, then passed to create_chunks.py

import whisper
import torch
import json

#  Explicitly set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = whisper.load_model("medium", device=device)  #  Pass device here and it uses gpu if available
print(f"Model on: {next(model.parameters()).device}") # not really necessary, just to confirm device

result = model.transcribe(
    audio="audios/sample.mp3",
    language="hi",
    task="translate",
    fp16 = False,
    # word_timestamps=False
)

# print(result)
chunks = []
for segment in result["segments"]:
    chunks.append({
        "start": segment["start"],
        "end": segment["end"],
        "text": segment["text"]
    })
print(chunks)

with open("sample_output.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=4)