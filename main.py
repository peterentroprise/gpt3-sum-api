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
import ffmpeg


class SynthesizedAudioInput(BaseModel):
    text: str
    voiceId: str

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

    def text_to_speech(voiceId, text):
        config = get_settings()
        url = "https://api.elevenlabs.io/v1/text-to-speech/%s" % (voiceId)
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

    text_to_speech(input.voiceId, input.text)
    return FileResponse("synthesizedAudio.mp3", media_type='audio/mpeg')


@app.post("/trainModel/")
async def train_model(file: UploadFile):
    name = file.filename

    print(name)

    with open(name, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    input_stream = ffmpeg.input(name)
    output_stream = ffmpeg.output(input_stream, "trainModelVideo.mp4")
    ffmpeg.run(output_stream, overwrite_output=True)
    video_clip = mp.VideoFileClip("trainModelVideo.mp4")
    video_clip.audio.write_audiofile(r"trainModelAudio.mp3")

    def voice_add(file_path, name):
        config = get_settings()
        url = "https://api.elevenlabs.io/v1/voices/add"

        payload={'name': name, 'labels': ''}

        files=[('files',(file_path,open(file_path,'rb'),'audio/mpeg'))]

        headers = {
        'accept': 'application/json',
        'xi-api-key': 'b80ce6ad1a3013b0e0eb0f159262a724'
        }

        response = requests.request("POST", url, headers=headers, data=payload, files=files)

        if response.status_code == 200:
            return response.json()
        else:
            print('Error: ' + str(response.status_code))
            # return 'Error: ' + str(response.status_code) 

    api_response = voice_add("trainModelAudio.mp3", name)

    return {"voiceId": api_response.get("voice_id")}

@app.post("/generateTranscript/")
async def generateTranscript(file: UploadFile):
    name = file.filename

    print(name)

    with open(name, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    input_stream = ffmpeg.input(name)
    output_stream = ffmpeg.output(input_stream, "trainModelVideo.mp4")
    ffmpeg.run(output_stream, overwrite_output=True)
    video_clip = mp.VideoFileClip("trainModelVideo.mp4")
    video_clip.audio.write_audiofile(r"trainModelAudio.mp3")

    def upload_audio(filename):
        def read_file(filename, chunk_size=5242880):
            with open(filename, 'rb') as _file:
                while True:
                    data = _file.read(chunk_size)
                    if not data:
                        break
                    yield data

        headers = {'authorization': "7fdf42ab12b54f909316cb2e2897788c"}
        response = requests.post('https://api.assemblyai.com/v2/upload',
                                headers=headers,
                                data=read_file(filename))
        print(response.request)

        return(response.json())
    
    def transcribe_audio(audio_url):
        endpoint = "https://api.assemblyai.com/v2/transcript"
        json = { "audio_url": audio_url }
        headers = {
            "authorization": "7fdf42ab12b54f909316cb2e2897788c",
        }
        response = requests.post(endpoint, json=json, headers=headers)
        print(response.json())

        return(response.json())


    audio_url = upload_audio("trainModelAudio.mp3")
    transcript_response = transcribe_audio(audio_url.get("upload_url"))
    print(transcript_response)
    return {"transcriptPollingId": transcript_response.get("id")}
 
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
 
