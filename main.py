from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseSettings
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import torch
import requests
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment
from scipy.spatial.distance import cdist
import moviepy.editor as mp
import shutil

class SynthesizedAudioInput(BaseModel):
    text: str

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

@app.post("/synthesizeAudio/")
async def synthesize_audio(input: SynthesizedAudioInput):
    def get_voices():
        config = get_settings()
        url = 'https://api.elevenlabs.io/v1/voices'
        headers = {'xi-api-key': "b80ce6ad1a3013b0e0eb0f159262a724"}

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            return response.json()
        else:
            return 'Error: ' + str(response.status_code)

    voices = get_voices()
    print(voices)

    def text_to_speech(text):
        config = get_settings()
        url = 'https://api.elevenlabs.io/v1/text-to-speech/WgvsZ0EcMaJGcVgMkzSc'
        headers = {'xi-api-key': "b80ce6ad1a3013b0e0eb0f159262a724", 'Accept': 'audio/mpeg'}
        body = {
            "text": text,
            "voice_settings": {
                "stability": 1,
                "similarity_boost": 1
            }
        }

        response = requests.post(url, headers=headers, json=body)

        if response.status_code == 200:
            with open('synthesizedAudio.mp3', 'wb') as f:
                f.write(response.content)
        else:
            print()
            return 'Error: ' + str(response.status_code)

    text_to_speech(input.text)
    return FileResponse("synthesizedAudio.mp3", media_type='audio/mpeg')


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):


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
 