# RAG-based-AI-Teaching-Assistant

How to use this RAG AI Teaching assistant on your own data

## Step 1 - Collect your videos
Move all your video files to the videos folder

## Step 2 - Convert to mp3
Convert all the video files to mp3 by running videos_to_mp3.py file

## Step 3 - Convert mp3 to json
Convert all the mp3 files to json chunks by running mp3_to_json_chunks.py

## Step 4 - Convert the json files to Vectors
Use the file preprocess_json_chunks.py to convert the json files to a dataframe with Embeddings and save it as a joblib inside joblib folder

## Step 5 - create incoming query and convert to embeddings and create prompt generarion and feed to LLM in deepseek-r1.py file
Read the joblib file and load it into the memory. Then create a relevant prompt as per the user query and feed it to the LLM






<!--
add the audio and video details why did in .gitignore
tell what is the project about

also important thing is add the commands aswell like  , whisper github command , ffmpeg install to path , pytorch+cuda and driver requirements and all ok.. for project to run you need it right.. so make it all documented

tell about whsiepr openai model used and in that used medium model , switched from small.. you can you use large-v2 (best one)
tell about faster-whisper used ( 4x faster and less vram usage)

tell about be careful about cuda.. tell best to downlaod cuda 12.9 (13 is latest , but still it is not quite compatible as its new release) , and graphic driver needs to be above 520 i suppose
, best to download 560 or the latest one (570) and download cudnn latest one of v9 also , as cudnn is required for deep learning.. so cuddn . cuda and driver must be compatible with one another
so like i told downlaod cuda 12.9 , driver 560 and cudnn v9.. it works well

also ensure gpu theres no high usage from other applications (if any auto start is there , terminate/disable those)


-->
