from fastapi import FastAPI
from pydantic import BaseModel

from .modules import *

BASE_DIR = "./media/"
__version__ = "0.1.0"

video_data_loader = VDL()

# transcription model
whisper_model_path = "tiny.en" #"large-v2", search over transcript only
mm_model_path = "openai/clip-vit-base-patch32"
HF_TOKEN = ""
low_mem = True
transcription_model = A2T(whisper_model_path, HF_TOKEN, low_mem)

# Embedding model
sm_model_path = 'Alibaba-NLP/gte-base-en-v1.5' # large 
sm_embedding_model = T2E(sm_model_path)
mm_embedding_model = VT2E(mm_model_path)

# vector database
vector_db = VDB(sm_embedding_model, mm_embedding_model)

# main function
@timeit
def get_text_and_embeds(yt_path):
  
  # load the video
  start = time.time()
  audio_path, video_path, id = video_data_loader.load_video(yt_path)
  print(f"Loaded video in: {time.time() - start} seconds")
  print("\n")

  # handle video embeds
  start = time.time()
  clip_embeddings, timesteps = video_data_loader.load_clip_embeddings(id, mm_embedding_model)
  print(f"Loaded clip embeddings in: {time.time() - start} seconds")
  print("\n")

  # grab the transcript
  start = time.time()
  transcript, transcript_embeds, scores = transcription_model.transcribe_and_align(audio_path, sm_embedding_model)
  print(f"Loaded transcript in: {time.time() - start} seconds")
  print("\n")

  # add transcript to the vector db
  vector_db.add_table(transcript['segments'], transcript_embeds, id, 'transcript')

  # add footage to the vector db
  vector_db.add_table(timesteps, clip_embeddings, id, 'frames')

  vector_db.update_most_recent(id)

  return transcript, transcript_embeds, scores


@timeit
# number of docs in final retrieval step
def search_video(query, table_cls='transcript', n=5, id=None):
  out_transcript, out_frames = vector_db.search_table(query, table_cls, n, id)
  return out_transcript, out_frames

app = FastAPI()

class VideoInput(BaseModel):
    link: str

class SearchQuery(BaseModel):
    query: str
    table_cls: str
    n: int = 5

class SearchOutput(BaseModel):
    start_time: float

class VideoOutput(BaseModel):
    transcript: dict
    embeds: list
    scores: list

@app.get("/")
def home():
    return {"health_check": "OK", "model_version": __version__}

@app.post("/add", response_model=VideoOutput)
async def add(yt_link: VideoInput):
  out = get_text_and_embeds(yt_link.link)
  return {"loader_output":out}

@app.post("/search", response_model=SearchOutput)
async def search(query: SearchQuery):
  start_time = search_video(query.query, query.table_cls, query.n)
  return {"start_time": start_time}
