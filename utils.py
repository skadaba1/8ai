import pandas as pd

def cluster_rows(df, threshold):
    
    # preprocess
    df = pd.DataFrame(df).reset_index(drop=True)
    df.sort_values(by=['start', 'end'], inplace=True)

    # Initialize the clusters
    clusters = []
    
    # Keep a reference to the current cluster
    
    # List to store clusters
        
    # Current cluster initialization
    current_cluster = {
        'start': df.loc[0, 'start'],
        'end': df.loc[0, 'end'],
        'text': df.loc[0, 'text'],
        'words': df.loc[0, 'words'],
        'source': df.loc[0, 'source'],
        'original_indices': df.index[0]
    }

    current_cluster = None
    
    # Iterate through the DataFrame
    for idx, row in df.iterrows():
      
      if(current_cluster is None):
        current_cluster = {
            'start': row['start'],
            'end': row['end'],
            'text': row['text'],
            'words': row['words'],
            'rank': idx,
            'source': row['source']
        }
      else:
        if(row['start'] - current_cluster['end'] <= threshold and row['source'] == current_cluster['source']):
          current_cluster['end'] = row['end']
          current_cluster['text'] += row['text']
          current_cluster['words'] += row['words']
          current_cluster['rank'] = min(idx, current_cluster['rank'])
          current_cluster['source'] = row['source']
        else:
          clusters.append(current_cluster)
          current_cluster = {
              'start': row['start'],
              'end': row['end'],
              'text': row['text'],
              'words': row['words'],
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