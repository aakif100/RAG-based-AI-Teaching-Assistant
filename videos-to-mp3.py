# converting videos to audios using ffmpeg
import os
import subprocess


files = os.listdir("videos")

for file in files:
    file_number = file.split("#")[1].split(".")[0]
    # print(file_number)
    # print(file)
    file_name = file.split(" _ ")[0]
    # print(file_name)
    subprocess.run(["ffmpeg","-i" ,f"videos/{file}",f"audios/{file_number}_{file_name}.mp3" ])







