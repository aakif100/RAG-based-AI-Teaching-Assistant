#  checking if the model is being loaded as there can be network restrictions ( and i have experienced it with firewalls bruhh)

# import whisper
# model = whisper.load_model("tiny")
# print("Model loaded successfully!")


import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version (PyTorch):", torch.version.cuda)
print("CUDA version (Driver):", torch.cuda.get_device_properties(0).major, ".", torch.cuda.get_device_properties(0).minor)



