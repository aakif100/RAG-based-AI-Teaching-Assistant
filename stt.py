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

import whisper
import torch

#  Explicitly set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = whisper.load_model("small", device=device)  #  Pass device here
print(f"Model on: {next(model.parameters()).device}")

result = model.transcribe(
    audio="audios/12_Exercise 1 - Pure HTML Media Player.mp3",
    language="hi",
    task="translate"
)

print(result["text"])





# ------------------------------------------------------------------------

