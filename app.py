from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import decord as de
import os

from model.modules import *
from model.modules import device as compute_device
import json


# constants
LMAX_FOR_GPU = 60
__version__ = "0.1.0"

# devices
ctx = de.cpu()
device = compute_device

# tokens
os.environ["OPENAI_API_KEY"] = ""
HF_TOKEN = ""


######################################################################################

################               The app core functionality             ################

######################################################################################

video_data_loader = VDL()

# transcription model
whisper_model_path = "tiny.en" #"large-v2", search over transcript only
mm_model_path = "openai/clip-vit-base-patch32"
low_mem = True
transcription_model = A2T(whisper_model_path, HF_TOKEN, low_mem)

# Embedding model
sm_model_path = 'BAAI/bge-small-en-v1.5' #'Alibaba-NLP/gte-base-en-v1.5', large 
sm_embedding_model = T2E(sm_model_path)
mm_embedding_model = VT2E(mm_model_path)

# vector database
vector_db = VDB(sm_embedding_model, mm_embedding_model)

# navigator
nav = Navigator()


# main functions
@timeit
def get_text_and_embeds(yt_path):
  
  # load the video
  id = search_for_url(yt_path, video_data_loader.lookup_table)
  if(id is None):
    start = time.time()
    audio_path, video_path, id, duration = video_data_loader.load_video(yt_path)
    print(f"Loaded video in: {time.time() - start} seconds")
    print("\n")

    # handle video embeds
    start = time.time()
    clip_embeddings, timesteps = video_data_loader.load_clip_embeddings(id, mm_embedding_model)
    print(f"Loaded clip embeddings in: {time.time() - start} seconds")
    print("\n")

  
    # if too long deal with video
    start = time.time()
    if(duration > LMAX_FOR_GPU*60):
      align = False
    else:
      align = True

   # grab the transcript
    transcript, transcript_embeds, scores = transcription_model.transcribe_and_align(audio_path, sm_embedding_model, align=align)
    print(f"Loaded transcript in: {time.time() - start} seconds")
    print("\n")

    # add transcript to the vector db
    vector_db.add_table(transcript['segments'], transcript_embeds, id, 'transcript')

    # add footage to the vector db
    vector_db.add_table(timesteps, clip_embeddings, id, 'frames')

    vector_db.update_most_recent(id)

    if os.path.exists("demofile.txt"):
      os.remove(video_data_loader.lookup_table[id]['audio'])
      os.remove(video_data_loader.lookup_table[id]['video'])
    else:
      print("The file does not exist")

    return transcript['segments'], time_util(duration)

  else:

    vector_db.update_most_recent(id)

    return vector_db.lookup_segments[id], video_data_loader.lookup_table[id]['duration']



@timeit
# number of docs in final retrieval step
def search_video(query, table_cls='transcript', n=25, id=None):
  out, source, _ = vector_db.search_table(query, table_cls, n, id)
  source = video_data_loader.lookup_table[source]['url']
  print(f"Source: {source}")
  gpt_answer = None

  if(_ is not None):
    gpt_answer = _

  # second argument is time threshold for clustering segments
  clustered = cluster_rows(out, table_cls, 5)
  nav.setup_nav(clustered['start'].values.tolist(), clustered['end'].values.tolist())
  start, end = clustered.iloc[0]['start'], clustered.iloc[0]['end']

  return start, end, source, gpt_answer

app = FastAPI()

# Configure CORS
origins = [
    "http://localhost:3000",  # The origin for your frontend
    "https://skadaba1.github.io",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],    # Allows all methods
    allow_headers=["*"],    # Allows all headers
)


class VideoInput(BaseModel):
    link: str

class SearchQuery(BaseModel):
    query: str
    table_cls: str
    n: int = 5

class SearchOutput(BaseModel):
    start_time: float
    end_time: float
    source: str
    meta: str
  
class NavOutput(BaseModel):
  start_time: float
  end_time: float

class VideoOutput(BaseModel):
    transcript: str
    duration: str

class DeletionSuccess(BaseModel):
    err: str

@app.get("/")
def home():
    return {"health_check": "OK", "model_version": __version__}

@app.post("/add", response_model=VideoOutput)
async def add(yt_link: VideoInput):
  transcript, duration = get_text_and_embeds(yt_link.link)
  out = json.dumps([{'start':segment['start'], 'end': segment['end'], 'text': segment['text']} for segment in transcript])
  return {"transcript":out, "duration":duration}

@app.post("/search", response_model=SearchOutput)
async def search(query: SearchQuery):
  start_time, end_time, source, _ = search_video(query.query, query.table_cls, query.n)

  if(_ is not None):
    gpt_answer = _
  else:
    gpt_answer = ""

  return {"start_time": start_time, "end_time":end_time, "source": source, "meta":gpt_answer}

@app.post("/delete", response_model=DeletionSuccess)
async def clear():
  try:
    vector_db.remove_table()
    video_data_loader.delete()
    return {"err":"Successfully cleared DB!"}
  except Exception as e:
    return {"err": str(e)}

@app.post("/next", response_model=NavOutput)
def next():
  start, end = nav.next()
  return {"start_time":start, "end_time":end}

@app.post("/prev", response_model=NavOutput)
def prev():
  start, end = nav.prev()
  return {"start_time":start, "end_time":end}