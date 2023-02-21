from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseSettings
from pydantic import BaseModel

import requests
import shutil
import os
import ffmpeg
import moviepy.editor as mp
# from scipy.spatial.distance import cdist


#import torch
# from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
# from pyannote.audio import Audio
# from pyannote.core import Segment

# model = PretrainedSpeakerEmbedding(
#     "speechbrain/spkrec-ecapa-voxceleb",
#     device=torch.device("cuda"))

class SynthesizedAudioInput(BaseModel):
    text: str
    voiceId: str
    stability: int
    similarity: int

class Settings(BaseSettings):
    eleven_labs_api_key: str
    aai_api_key: str
    class Config:
        env_file = ".env"

def get_settings():
    return Settings()



app = FastAPI()

origins = [
    "https://gpt3-sum.vercel.app",
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    config = get_settings()
    print(config.eleven_labs_api_key)
    return {"message": "Hello Universe."}


@app.post("/deleteAllVoices/")
async def delete_all_voices():
    def get_voices():
        config = get_settings()
        url = 'https://api.elevenlabs.io/v1/voices'
        headers = {'xi-api-key': config.eleven_labs_api_key}

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            return response.json()
        else:
            return 'Error: ' + str(response.status_code)

    def delete_voice(voice_id):
            config = get_settings()
            url = 'https://api.elevenlabs.io/v1/voices/' + voice_id
            headers = {'xi-api-key': config.eleven_labs_api_key}

            response = requests.delete(url, headers=headers)

            if response.status_code == 200:
                return response.json()
            else:
                return 'Error: ' + str(response.status_code)

    voices = get_voices()
    voice_list = voices["voices"]
    for voice in voice_list:
        print(voice)
        if(voice["category"] == "cloned"):
            delete_voice(voice["voice_id"])

    return voices

@app.get("/getAllVoices/")
async def get_all_voices():
    def get_voices():
        config = get_settings()
        url = 'https://api.elevenlabs.io/v1/voices'
        headers = {'xi-api-key': config.eleven_labs_api_key}

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            return response.json()
        else:
            return 'Error: ' + str(response.status_code)

    voices = get_voices()
    print(voices)

    return voices


@app.post("/synthesizeAudio/")
async def synthesize_audio(input: SynthesizedAudioInput):
    def text_to_speech(voiceId, text, stability, similarity):
        config = get_settings()
        url = "https://api.elevenlabs.io/v1/text-to-speech/%s" % (voiceId)
        headers = {'xi-api-key': config.eleven_labs_api_key, 'Accept': 'audio/mpeg'}
        body = {
            "text": text,
            "voice_settings": {
                "stability": stability,
                "similarity_boost": similarity
            }
        }

        response = requests.post(url, headers=headers, json=body)

        if response.status_code == 200:
            with open('synthesizedAudio.mp3', 'wb') as f:
                f.write(response.content)
        else:
            print()
            return 'Error: ' + str(response.status_code)

    text_to_speech(input.voiceId, input.text, input.stability, input.similarity)
    return FileResponse("synthesizedAudio.mp3", media_type='audio/mpeg')


@app.post("/trainModel/")
async def train_model(file: UploadFile):
    original_name = file.filename

    name_split = os.path.splitext(original_name)
    file_name = "trainModelVideo"
    file_extension = name_split[1]

    modified_name = file_name + file_extension

    with open(modified_name, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    if(file_extension == ".webm"):
        mp4_name = file_name + ".mp4"
        input_stream = ffmpeg.input(modified_name)
        output_stream = ffmpeg.output(input_stream, mp4_name)
        ffmpeg.run(output_stream, overwrite_output=True)
        os.remove(modified_name)
        modified_name = mp4_name
       
    video_clip = mp.VideoFileClip(modified_name)
    video_clip.audio.write_audiofile(r"trainModelAudio.mp3")
    os.remove(modified_name)

    def voice_add(file_path, original_name):
        config = get_settings()
        url = "https://api.elevenlabs.io/v1/voices/add"

        payload={'name': original_name, 'labels': ''}

        files=[('files',(file_path,open(file_path,'rb'),'audio/mpeg'))]

        headers = {
        'accept': 'application/json',
        'xi-api-key': config.eleven_labs_api_key
        }

        response = requests.request("POST", url, headers=headers, data=payload, files=files)

        if response.status_code == 200:
            return response.json()
        else:
            print('Error: ' + str(response.status_code))

    api_response = voice_add("trainModelAudio.mp3", original_name)
    os.remove("trainModelAudio.mp3")

    return {"voiceId": api_response.get("voice_id")}

@app.post("/generateTranscript/")
async def generateTranscript(file: UploadFile):
    config = get_settings()
    original_name = file.filename

    name_split = os.path.splitext(original_name)
    file_name = "trainModelVideo"
    file_extension = name_split[1]

    modified_name = file_name + file_extension

    with open(modified_name, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    if(file_extension == ".webm"):
        mp4_name = file_name + ".mp4"
        input_stream = ffmpeg.input(modified_name)
        output_stream = ffmpeg.output(input_stream, mp4_name)
        ffmpeg.run(output_stream, overwrite_output=True)
        os.remove(modified_name)
        modified_name = mp4_name


    video_clip = mp.VideoFileClip(modified_name)
    video_clip.audio.write_audiofile(r"trainModelAudio.mp3")
    os.remove(modified_name)

    def upload_audio(filename):
        def read_file(filename, chunk_size=5242880):
            with open(filename, 'rb') as _file:
                while True:
                    data = _file.read(chunk_size)
                    if not data:
                        break
                    yield data

        headers = {'authorization': config.aai_api_key}
        response = requests.post('https://api.assemblyai.com/v2/upload',
                                headers=headers,
                                data=read_file(filename))
        print(response.request)

        return(response.json())
    
    def transcribe_audio(audio_url):
        endpoint = "https://api.assemblyai.com/v2/transcript"
        json = { "audio_url": audio_url, "speaker_labels": True, "sentiment_analysis": True }
        headers = {
            "authorization": config.aai_api_key,
        }
        response = requests.post(endpoint, json=json, headers=headers)
        print(response.json())

        return(response.json())


    audio_url = upload_audio("trainModelAudio.mp3")
    transcript_response = transcribe_audio(audio_url.get("upload_url"))
    os.remove("trainModelAudio.mp3")
    print(transcript_response)
    return {"transcriptPollingId": transcript_response.get("id")}
 
# @app.post("/uploadfile/")
# async def create_upload_file(file: UploadFile):

#     with open("tempVideo.webm", "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)

#     video_clip = mp.VideoFileClip("tempVideo.webm")
#     video_clip.audio.write_audiofile(r"tempAudio.wav")

#     audio = Audio(sample_rate=16000, mono=True)

#     speaker1 = Segment(0., 1.)
#     waveform1, sample_rate = audio.crop("tempAudio.wav", speaker1)
#     embedding1 = model(waveform1[None])

#     speaker2 = Segment(1., 2.)
#     waveform2, sample_rate = audio.crop("tempAudio.wav", speaker2)
#     embedding2 = model(waveform2[None])

#     distance = cdist(embedding1, embedding2, metric="cosine")
#     distanceNumber = distance[0][0]

#     return {"filename": file.filename, "distance": distanceNumber}
 
