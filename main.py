from fastapi import FastAPI, File, UploadFile
from pydantic import BaseSettings
from fastapi.middleware.cors import CORSMiddleware
import torch
import requests
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment
from scipy.spatial.distance import cdist
import moviepy.editor as mp
import shutil

class Settings(BaseSettings):
    eleven_labs_api_key: str
    class Config:
        env_file = ".env"

def get_settings():
    return Settings()

model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb",
    device=torch.device("cuda"))

app = FastAPI()

origins = [

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
    return {"message": "Hello Universe."}


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):

    def get_voices():
        config = get_settings()
        url = 'https://api.elevenlabs.io/v1/voices'
        params = {'xi-api-key': config.eleven_labs_api_key}

        response = requests.get(url, params=params)

        if response.status_code == 200:
            return response.json()
        else:
            return 'Error: ' + str(response.status_code)

    voices = get_voices()
    print(voices)

    def text_to_speech():
        config = get_settings()
        url = 'https://api.elevenlabs.io/v1/voices'
        params = {'xi-api-key': config.eleven_labs_api_key, "voice_id": "21m00Tcm4TlvDq8ikWAM"}
        body = {
            "text": "hello universe take me there",
            "voice_settings": {
                "stability": 0,
                "similarity_boost": 0
            }
        }

        response = requests.get(url, params=params, json=body)

        if response.status_code == 200:
            return response.json()
        else:
            print(response.json())
            return 'Error: ' + str(response.status_code)


    audio = text_to_speech()
    print(audio)

    with open("tempVideo.webm", "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    video_clip = mp.VideoFileClip("tempVideo.webm")
    video_clip.audio.write_audiofile(r"tempAudio.wav")

    audio = Audio(sample_rate=16000, mono=True)

    speaker1 = Segment(0., 1.)
    waveform1, sample_rate = audio.crop("tempAudio.wav", speaker1)
    embedding1 = model(waveform1[None])

    speaker2 = Segment(1., 2.)
    waveform2, sample_rate = audio.crop("tempAudio.wav", speaker2)
    embedding2 = model(waveform2[None])

    distance = cdist(embedding1, embedding2, metric="cosine")
    distanceNumber = distance[0][0]

    return {"filename": file.filename, "distance": distanceNumber}