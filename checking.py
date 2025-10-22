#  checking if the model is being loaded as there can be network restrictions ( and i have experienced it with firewalls bruhh)

import whisper
model = whisper.load_model("tiny")
print("Model loaded successfully!")
