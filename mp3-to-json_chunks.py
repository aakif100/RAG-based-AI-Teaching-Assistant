# important step this is as it does creation of chunks for vector db ingestion and this will take time depending on audio length

import whisper
import torch
import json
import os

#  Explicitly set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = whisper.load_model("medium", device=device)  #  Pass device here and it uses gpu if available
print(f"Model on: {next(model.parameters()).device}") # not really necessary, just to confirm device


audios = os.listdir("audios")

for audio in audios:
    if "_" in audio:
        number = audio.split("_")[0]
        title = audio.split("_")[1][:-4] # removing .mp3
        # print(f"Audio Number: {number}, Title: {title}")
        result = model.transcribe(
            audio=f"audios/{audio}",
            # audio=f"audios/sample.mp3",
            language="hi",
            task="translate",
            fp16 = False,
            # word_timestamps=False  
        )
        chunks = []
        for segment in result["segments"]:
            chunks.append({
                "number": number,
                "title": title,
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            })
        
        chunks_with_metadata = { "chunks":chunks , "text": result["text"]}

        with open(f"jsons/{audio}.json", "w", encoding="utf-8") as f:
            json.dump(chunks_with_metadata, f, ensure_ascii=False, indent=4)