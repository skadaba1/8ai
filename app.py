from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# from .modules import * ##################################################

import pytube

import os
from openai import OpenAI
from functools import wraps
import whisperx

import torch
import time
import json

import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
import numpy as np
import uuid
from numba import cuda

import ffmpeg
from PIL import Image

from transformers import AutoTokenizer, CLIPTextModelWithProjection
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from PIL import Image

# from .utils import *  #######################################################
import pandas as pd
import decord as de

ctx = de.cpu()
os.environ["OPENAI_API_KEY"] = ""
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def cluster_rows(df, table_cls, threshold):

    # preprocess
    df = pd.DataFrame(df).reset_index(drop=True)
    df.sort_values(by=['start', 'end'], inplace=True)

    # Initialize the clusters
    clusters = []

    current_cluster = None

    # Iterate through the DataFrame
    for idx, row in df.iterrows():

      if(current_cluster is None):
        if(table_cls == 'transcript'):
          current_cluster = {
              'start': row['start'],
              'end': row['end'],
              'text': row['text'],
              'words': row['words'],
              'rank': idx,
              'source': row['source']
          }
        else:
          current_cluster = {
              'start': row['start'],
              'end': row['end'],
              'source': row['source'],
              'rank': idx,
          }
      else:
        if(row['start'] - current_cluster['end'] <= threshold and row['source'] == current_cluster['source']):
          current_cluster['end'] = row['end']
          if(table_cls == 'transcript'):
            current_cluster['text'] += row['text']
            current_cluster['words'] += row['words']
          current_cluster['rank'] = min(idx, current_cluster['rank'])
          current_cluster['source'] = row['source']
        else:
          clusters.append(current_cluster)
          if(table_cls == 'transcript'):
            current_cluster = {
                'start': row['start'],
                'end': row['end'],
                'text': row['text'],
                'words': row['words'],
                'source': row['source'],
                'rank': idx
            }
          else:
            current_cluster = {
                'start': row['start'],
                'end': row['end'],
                'source': row['source'],
                'rank': idx
            }

    clusters.append(current_cluster)

    # Create a new DataFrame from clusters
    clustered_df = pd.DataFrame(clusters).sort_values(by=['rank'])
    clustered_df.reset_index(drop=True, inplace=True)

    return clustered_df


def time_util(total_seconds):
  
  hours = int(total_seconds // 3600)
  remaining_seconds = total_seconds % 3600
  minutes = int(remaining_seconds // 60)
  seconds = remaining_seconds % 60
  formatted_time = f"{hours:02}:{minutes:02}:{seconds:.2f}"

  return formatted_time

def search_for_url(url, obj):
    for id, details in obj.items():
        # Check if the url matches the given video_url
        if details['url'] == url:
            return id
    return None

def combine_tables(dataframes):

  # Concatenate the DataFrames with keys as a multi-index
  combined_df = pd.concat(dataframes.values())

  # Reset the index to create a new column with the key
  combined_df.reset_index(level=0, inplace=True)

  return combined_df

def extract_words_around_time(dataframe, start_time, n):
    # Define the time range
    time_start = start_time - n
    time_end = start_time + n

    # List to store the extracted words
    extracted_words = []

    # Loop through the dataframe rows to find segments that overlap with the given time range
    for index, row in dataframe.iterrows():
        # Check if there's an overlap between the row's segment and the given time range
        if row['end'] >= time_start and row['start'] <= time_end:
            # Loop through the words in this segment
            for word_info in row['words']:
                if('start' in word_info):
                  # Check if the word falls within the given time range
                  if word_info['start'] >= time_start and word_info['end'] <= time_end:
                      extracted_words.append(word_info)

    # Sort the extracted words by their start times to maintain order
    extracted_words.sort(key=lambda x: x['start'])
    paragraph = [word_info['word'] for word_info in extracted_words]
    return " ".join(paragraph)

    ########################################################################################


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Record the start time
        start_time = time.time()

        # Call the function
        result = func(*args, **kwargs)

        # Record the end time
        end_time = time.time()

        # Calculate the time it took to execute the function
        elapsed_time = end_time - start_time

        # Print the time taken
        print(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds")

        # Return the function's result
        return result
    return wrapper


# Video Loader 
class VDL():

  def __init__(self, model_name="openai/clip-vit-base-patch32"):
    self.lookup_table = dict()

  def _get_scaled_size_(self, width, height):
      target_width = 224
      w_percent = (target_width / float(width))
      h_size = int((float(height) * float(w_percent)))
      return target_width, h_size

  @timeit
  def _get_frames_(self, video_path, num_frames=500):
      vr = de.VideoReader(video_path)
      fps = vr.get_avg_fps()
      duration = vr.get_frame_timestamp([-1]).flatten()[0]

      selected_frames = np.linspace(0, int(duration*fps), num=num_frames, endpoint=False, dtype=int).tolist()
      timestamps_temp = vr.get_frame_timestamp(selected_frames).flatten().tolist()[::2]
    
      #[0, 197, 394, 591, 789, 986, 1183, 1381, 1578, 1775]
      frames = []
      timestamps = []
      for i, t in enumerate(selected_frames):
          try:
              frame = vr.get_batch([selected_frames[i]])
              frames.append(frame)
              timestamps.append(timestamps_temp[i])
          except:
              pass
      frames = [i.asnumpy()[0,:,:] for i in frames]
      print(len(frames), len(timestamps))
      return frames, np.array(timestamps)
    
      

  @timeit
  def _get_embeddings_(self, frames, clip_model):
      # Convert frames to a list of PIL Images
      images = [Image.fromarray(frame) for frame in frames]
      
      # Batch embed the frames
      embeddings = clip_model.embed_image(images)
      
      # Convert the embeddings to a list of NumPy arrays
      embeddings_list = embeddings.detach().cpu().numpy().tolist()

      return embeddings_list
  
  def _extract_clip(self, video_path, clip_model):
      frames, timesteps = self._get_frames_(video_path)
      embeddings = self._get_embeddings_(frames, clip_model)
      return embeddings, timesteps
      
    # Function to download audio and extract clips of length n seconds
  def load_video(self, video_url, output_directory='./'):
      try:

          # Initialize a YouTube object with the video URL
          yt = pytube.YouTube(video_url)

          # Get the audio-only stream with the highest bitrate
          audio_stream = yt.streams.filter(only_audio=True).order_by('abr').last()
          video_stream = yt.streams.filter(only_video=True).order_by('resolution').last()

          # Download the audio and video streams
          audio_path = audio_stream.download(output_path=output_directory+'audio/')
          video_path = video_stream.download(output_path=output_directory+'video/')
          id = uuid.uuid4()

          self.lookup_table[id] = {'audio':audio_path, 'video':video_path, 'url':video_url, 'duration':yt.length}

          return audio_path, video_path, id, yt.length

      except Exception as e:
          print(f"An error occurred: {e}")
          return None
  
  @timeit
  def load_clip_embeddings(self, id, clip_model):

      # Extract CLIP embeddings
      video_path = self.lookup_table[id]['video']

      embeddings, timesteps = self._extract_clip(video_path, clip_model)
      self.lookup_table[id]['embeds'] = embeddings
      self.lookup_table[id]['embedding_timesteps'] = timesteps
      return embeddings, timesteps
  
  def get_audio_path(self, id):
    return self.lookup_table[id]['audio']
  
  def get_video_path(self, id):
    return self.lookup_table[id]['video']
  
  def get_embeddings(self, id):
    return self.lookup_table[id]['embeddings']
  
  def delete(self):
    self.lookup_table = dict()
  

# Audio to Text

class A2T():
  def __init__(self, model_name, HF_TOKEN=None, low_mem=True, batch_size=4):

    if(not HF_TOKEN):
      raise Exception("You must provide a valid Hugging Face authentication token!")
    else:
      self.hf_token = HF_TOKEN

    self.model = None
    self.model_a = None
    self.metadata = None
    self.diarize_model = None

    # config
    self.batch_size = batch_size
    self.device = device
    print(f'Device: {self.device}')


    self._setup_(model_name, low_mem)

  def _setup_(self, model_name, low_mem):
    compute_type = "int8" if low_mem else "float16"
    self.model = whisperx.load_model(model_name, self.device, compute_type=compute_type)

    # lazy loader, assumes all videos in English
    if(self.model_a is None):
      self.model_a, self.metadata = whisperx.load_align_model(language_code="en", device=self.device) 
    # save model to local path (optional)

     # lazy loader
    if(self.diarize_model is None):
      self.diarize_model = whisperx.DiarizationPipeline(use_auth_token=self.hf_token, device=self.device)

    # model_dir = "/path/"
    # model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

  @timeit
  def transcribe_and_align(self, audio_file, embedding_model=None, diarize=False):

    # 1. Transcribe audio file
    audio = whisperx.load_audio(audio_file)
    result = self.model.transcribe(audio, batch_size=self.batch_size)

    # 2. Align whisper output
                               
    result = whisperx.align(result["segments"], self.model_a, self.metadata, audio, self.device, return_char_alignments=False)

    if(diarize):
      # 3. Assign speaker labels
      diarize_segments = self.diarize_model(audio)
      # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

      result = whisperx.assign_word_speakers(diarize_segments, result)

    if(not embedding_model):
      raise Exception("You must provide an embedding model!")
    else:
      input_texts = [segment['text'] for segment in result['segments']]
      embeds, scores = embedding_model.embed(input_texts)
    return result, embeds, scores


# Requires transformers>=4.36.0

# text to embedding
class T2E():

  def __init__(self, model_name, preprocess=False):
    self.model = None
    self.tokenizer = None
    self.preprocess = preprocess

    self._setup_(model_name)

  def _setup_(self, model_name):
    self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)

  def _preprocess_(self, last_hidden_state):
    if(self.preprocess):
      embeddings = F.normalize(embeddings, p=2, dim=1)
    else:
      return last_hidden_state

  def _compute_scores(self, embeddings):
    return ((embeddings[:1] @ embeddings[1:].T) * 100).tolist()

  @timeit
  def embed(self, input_texts):
    batch_dict = self.tokenizer(input_texts, max_length=8192, padding=True, return_tensors='pt')
    inputs = {key: value.to(device) for key, value in batch_dict.items()}
    outputs = self.model(**inputs)
    embeddings = self._preprocess_(outputs.last_hidden_state[:,0])
    scores = self._compute_scores(embeddings)
    return embeddings, scores

# text-vision MM embedding
class VT2E():
  def __init__(self, model_name):
    self.image_model = None
    self.image_tokenizer = None

    self.text_model = None
    self.text_tokenizer = None
    

    self._setup_(model_name)

  def _setup_(self, model_name):

    self.image_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(device)
    self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

    self.text_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(device)
    self.text_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
  
  def embed_text(self, input_texts):
    inputs = self.text_tokenizer(input_texts, padding=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    outputs = self.text_model(**inputs)
    text_embeds = outputs.text_embeds
    return text_embeds
  
  def embed_image(self, images):
    inputs = self.image_processor(images, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    outputs = self.image_model(**inputs)
    image_embeds = outputs.image_embeds
    return image_embeds


# vector data base
class VDB():
  def __init__(self, text_embedding_model, image_embedding_model):

    self.lookup_transcript = dict()
    self.lookup_frames = dict()
    self.lookup_segments = dict()

    self.text_embedding_model = text_embedding_model
    self.image_embedding_model = image_embedding_model

    self.most_recent = None
    self.client = None

    self._setup_()
  
  def _setup_(self):
    self.client = OpenAI()

  def add_table(self, segments, embeds, id, table_cls):

    if(table_cls == 'transcript'):
      self.lookup_segments[id] = segments
      transcript = "\n".join([seg['text'] for seg in segments])
      for entry, embed in zip(segments, embeds):
        entry['embed'] = embed.detach().cpu().numpy().flatten()
      df = pd.DataFrame(segments)
      df['source'] = id
      self.lookup_transcript[id] = df

    elif(table_cls == 'frames'):

      # Create a pandas DataFrame with two columns
      data = {
          "start": segments.flatten(),  # Convert (50, 1) to (50,) for a flat column
          "end":segments.flatten(),
          "embed": embeds
      }

      # Create the DataFrame
      df = pd.DataFrame(data)
      df['source'] = id
      self.lookup_frames[id] = df
    else:
      raise Exception("Invalid table type!")
  
  def remove_table(self, id=None):
    if(id):
      del self.lookup_transcript[id]
      del self.lookup_frames[id]
      del self.lookup_segments[id]
    else:
      self.lookup_transcript = dict()
      self.lookup_frames = dict()
      self.lookup_segments = dict()

  def get_table(self, id, table_cls):

    if(table_cls == 'transcript'):

      if(id is not None):
        df = self.lookup_transcript[id]
        return df
      else:
        return combine_tables(self.lookup_transcript)

    elif(table_cls == 'frames'):

      if(id is not None):
        df = self.lookup_frames[id]
        return df
      else:
        return combine_tables(self.lookup_frames)

    else:
      raise Exception("Invalid table type!")
    
  
  def _openai_pipe_(self, query, docs, id=None, gpt_model_name='gpt-3.5-turbo-0125'):

    system_prompt = f"You are a helpful assistant which answers question about a set of video transcripts. \
                      Given the following retrieved chunks of a transcript: {docs} of a video, answer the user's \
                      question in no more than 1 sentence."
    user_prompt = f"User's question: {query}"
    messages = [
        {'role':'system', 'content':system_prompt}, 
         {'role':'user', 'content':user_prompt}
    ]
    temperature=0.2
    max_tokens=256

    response = self.client.chat.completions.create(
      model=gpt_model_name,
      messages=messages,
      temperature=temperature,
      max_tokens=max_tokens,
    )
    return response.choices[0].message.content

  def _get_segments_(self, start_times, dataframe):
      # Initialize an empty dataframe to store matching rows
      matching_rows = pd.DataFrame()

      # Iterate through the list of start_times
      for start_time in start_times:
          # Find rows where start_time is between "start" and "end"
          filtered_rows = dataframe[
              (dataframe['start'] <= start_time) & (dataframe['end'] >= start_time)
          ]

          # Append the matching rows to the result dataframe
          matching_rows = matching_rows.append(filtered_rows, ignore_index=True)

      return matching_rows
    
  def _exec_search_(self, embed, table, n=5):

    all_embeddings = np.stack(table['embed'].values)
    similarities = cosine_similarity(all_embeddings, [embed])[:,0]
    top_indices = np.argsort(-similarities)[:n]  # Argsort in descending order and slice the top n
    top_rows = table.iloc[top_indices]

    return top_rows
    
  def search_table(self, query, table_cls, n=5, id=None):

    if(table_cls == 'transcript'):

      table = self.get_table(id, table_cls)
      query = query if type(query) == list else [query]
      normalized_query_embed = self.text_embedding_model.embed(query)[0].detach().cpu().numpy().flatten()

      top_rows = self._exec_search_(normalized_query_embed, table, n=20) # harcoded 

      # Get the top n rows from the DataFrame
      docs = ""
      for i, row in top_rows.iterrows():
        words = extract_words_around_time(self.get_table(row['source'], table_cls), row['start'], 10) # 10 seconds around each word
        docs += words + "\n"

      query = self._openai_pipe_(query, docs)
      print(f"GPT-reworded query: {query}")
      normalized_query_embed = self.text_embedding_model.embed(query)[0].detach().cpu().numpy().flatten()

      top_rows = self._exec_search_(normalized_query_embed, table, n=n)  

      out = top_rows.drop(columns=['embed'], axis=1)
      source = out['source'].values[0]

      return out, source, query

    elif(table_cls == 'frames'):

      table = self.get_table(id, table_cls)
      query = query if type(query) == list else [query]
      query_embed = self.image_embedding_model.embed_text(query)[0].detach().cpu().numpy().flatten()

      # Get the top n rows from the DataFrame
      top_rows = self._exec_search_(query_embed, table, n)

      #start_times = top_rows['start'].values
      #source = top_rows['source'].values[0]
      
      #top_rows_transcript = self._get_segments_(start_times, self.get_table(source, 'transcript'))

      out_frames = top_rows.drop(columns=['embed'], axis=1)
      #out_transcript = top_rows_transcript.drop(columns=['embed'], axis=1)
      source = out_frames['source'].values[0]

      return out_frames, source, None

    else:

      raise Exception("Invalid table type!")

  def update_most_recent(self, id):
    self.most_recent = id

class Navigator():

  def __init__(self):

    self.start_times = None
    self.end_times = None
    self.index = None
    self.length = None
  
  def setup_nav(self, start_times, end_times):
    self.start_times = start_times
    self.end_times = end_times
    self.index = 0
    self.length = len(start_times)
  
  def next(self):
    self.index = (self.index + 1) % self.length
    return self.start_times[self.index], self.end_times[self.index]
  
  def prev(self):
    self.index = (self.index - 1) % self.length
    return self.start_times[self.index], self.end_times[self.index]


##################################################################################

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
sm_model_path = 'BAAI/bge-small-en-v1.5' #'Alibaba-NLP/gte-base-en-v1.5', large 
sm_embedding_model = T2E(sm_model_path)
mm_embedding_model = VT2E(mm_model_path)

# vector database
vector_db = VDB(sm_embedding_model, mm_embedding_model)

# navigator
nav = Navigator()

##################################################################################

# main function
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

    if os.path.exists("demofile.txt"):
      os.remove(video_data_loader.lookup_table[id]['audio'])
      os.remove(video_data_loader.lookup_table[id]['video'])
    else:
      print("The file does not exist")

    # device = cuda.get_current_device()
    # device.reset()

    return transcript['segments'], time_util(duration)

  else:

    vector_db.update_most_recent(id)

    return vector_db.lookup_segments[id], video_data_loader.lookup_table[id]['duration']



@timeit
# number of docs in final retrieval step
def search_video(query, table_cls='transcript', n=5, id=None):
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

  # device = cuda.get_current_device()
  # device.reset()

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