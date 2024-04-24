
import pytube

import os
from openai import OpenAI
from functools import wraps
import whisperx

import torch
import time

import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
import numpy as np
import uuid

import ffmpeg
from PIL import Image

from transformers import AutoTokenizer, CLIPTextModelWithProjection
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from PIL import Image

from .utils import * 

os.environ["OPENAI_API_KEY"] = ""


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
  def _get_frames_(self, video_path):

      probe = ffmpeg.probe(video_path, threads=1)
      video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
      width, height = self._get_scaled_size_(int(video_info['width']), int(video_info['height']))

      duration = float(probe['format']['duration'])
      fps = eval(video_info['r_frame_rate'])  # Convert to float

      out, err = (
            ffmpeg
            .input(video_path, threads=1)
            .filter('scale', width, height)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, quiet=True)
        )
      frames = (
          np
          .frombuffer(out, np.uint8)
          .reshape([-1, height, width, 3])
      )

      # Number of frames in the video
      total_frames = frames.shape[0]
      
      # Define the number of evenly spaced indexes
      n = 100  # You can change this to the desired number of indexes

      # Create evenly spaced indexes
      indexes = np.linspace(0, total_frames - 1, num=n, dtype=int)
      timestamps = indexes / fps

      #indexes = np.random.randint(frames.shape[0], size=10)
      return [ frame for frame in frames[indexes, :] ], timestamps
    
      

  @timeit
  def _get_embeddings_(self, frames, clip_model):
      # Convert frames to a list of PIL Images
      images = [Image.fromarray(frame) for frame in frames]
      
      # Batch embed the frames
      embeddings = clip_model.embed_image(images)
      
      # Convert the embeddings to a list of NumPy arrays
      embeddings_list = embeddings.detach().numpy().tolist()

      return embeddings_list
  
  def _extract_clip(self, video_path, clip_model):
      frames, timesteps = self._get_frames_(video_path)
      embeddings = self._get_embeddings_(frames, clip_model)
      return embeddings, timesteps
      
    # Function to download audio and extract clips of length n seconds
  def load_video(self, video_url, output_directory='/content/'):
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

          self.lookup_table[id] = {'audio':audio_path, 'video':video_path}

          return audio_path, video_path, id

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
  

# Audio to Text

class A2T():
  def __init__(self, model_name, HF_TOKEN=None, low_mem=True, batch_size=16):

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
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
    self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
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
    outputs = self.model(**batch_dict)
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

    self.image_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
    self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

    self.text_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
    self.text_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
  
  def embed_text(self, input_texts):
    inputs = self.text_tokenizer(input_texts, padding=True, return_tensors="pt")

    outputs = self.text_model(**inputs)
    text_embeds = outputs.text_embeds
    return text_embeds
  
  def embed_image(self, images):
    inputs = self.image_processor(images, return_tensors="pt")
    outputs = self.image_model(**inputs)
    image_embeds = outputs.image_embeds
    return image_embeds


# vector data base
class VDB():
  def __init__(self, text_embedding_model, image_embedding_model):

    self.lookup_transcript = dict()
    self.lookup_frames = dict()

    self.text_embedding_model = text_embedding_model
    self.image_embedding_model = image_embedding_model

    self.most_recent = None
    self.client = None

    self._setup_()
  
  def _setup_(self):
    self.client = OpenAI()

  def add_table(self, segments, embeds, id, table_cls):
    if(table_cls == 'transcript'):
      transcript = "\n".join([seg['text'] for seg in segments])
      for entry, embed in zip(segments, embeds):
        entry['embed'] = embed.detach().numpy().flatten()
      df = pd.DataFrame(segments)
      df['source'] = id
      self.lookup_transcript[id] = df

    elif(table_cls == 'frames'):

      # Create a pandas DataFrame with two columns
      data = {
          "start": segments.flatten(),  # Convert (50, 1) to (50,) for a flat column
          "embed": embeds
      }

      # Create the DataFrame
      df = pd.DataFrame(data)
      df['source'] = id
      self.lookup_frames[id] = df
    else:
      raise Exception("Invalid table type!")
  
  def remove_table(self, id):
    del self.lookup_transcript[id]
    del self.lookup_frames[id]

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
      normalized_query_embed = self.text_embedding_model.embed(query)[0].detach().numpy().flatten()

      top_rows = self._exec_search_(normalized_query_embed, table, n=20) # harcoded 

      # Get the top n rows from the DataFrame
      docs = ""
      for i, row in top_rows.iterrows():
        words = extract_words_around_time(self.get_table(row['source'], table_cls), row['start'], 10) # 10 seconds around each word
        docs += words + "\n"

      query = self._openai_pipe_(query, docs)
      print(f"GPT-reworded query: {query}")
      normalized_query_embed = self.text_embedding_model.embed(query)[0].detach().numpy().flatten()

      top_rows = self._exec_search_(normalized_query_embed, table, n=n)  
      _ = None

      out = top_rows.drop(columns=['embed'], axis=1)
      return out, _

    elif(table_cls == 'frames'):

      table = self.get_table(id, table_cls)
      query = query if type(query) == list else [query]
      query_embed = self.image_embedding_model.embed_text(query)[0].detach().numpy().flatten()

      # Get the top n rows from the DataFrame
      top_rows = self._exec_search_(query_embed, table, n)

      start_times = top_rows['start'].values
      
      top_rows_transcript = self._get_segments_(start_times, self.get_table(id, 'transcript'))

      out_frames = top_rows.drop(columns=['embed'], axis=1)
      out_transcript = top_rows_transcript.drop(columns=['embed'], axis=1)
      return out_transcript, out_frames

    else:

      raise Exception("Invalid table type!")

  def update_most_recent(self, id):
    self.most_recent = id
