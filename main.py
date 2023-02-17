from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import torch
import io
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment
from scipy.spatial.distance import cdist

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
    file_contents = await file.read()
    temp_file = io.BytesIO(file_contents)
    temp_file.seek(0)

    audio = Audio(sample_rate=16000, mono=True)

    speaker1 = Segment(0., 1.)
    waveform1, sample_rate = audio.crop(temp_file, speaker1)
    embedding1 = model(waveform1[None])

    speaker2 = Segment(1., 2.)
    waveform2, sample_rate = audio.crop(temp_file, speaker2)
    embedding2 = model(waveform2[None])

    distance = cdist(embedding1, embedding2, metric="cosine")
    print(distance)

    return {"filename": file.filename}