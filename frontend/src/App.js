import React, { useState, useEffect } from 'react';
import YouTube from 'react-youtube';
import styled from 'styled-components';
import axios from 'axios';

const InputContainer = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px;
  margin: 10px 0;
  border: 1px solid #ccc;
  border-radius: 8px;
  background-color: #f9f9f9;

  input {
    flex: 1;
    padding: 8px;
    margin-right: 10px;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-size: 16px;
  }

  select {
    padding: 8px;
    margin-right: 10px;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-size: 16px;
    cursor: pointer;

    option {
      padding: 8px;
      background: white;
    }
  }

  button {
    padding: 8px 16px;
    background-color: #007BFF;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 16px;
    &:hover {
      background-color: #0056b3;
    }
  }
`;

const ButtonContainer = styled.div`

  button {
    padding: 8px 16px;
    background-color: #007BFF;
    margin: 20px;
    margin-left: 90%;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    right: 0px;            

    font-size: 16px;

    &:hover {
      background-color: #0056b3;
    }
  }
`;

// Styled component for the outer collapsible container
const CollapsibleContainer = styled.div`
  border: 1px solid #ccc;
  border-radius: 8px;
  background-color: #f9f9f9;
  padding: 10px;
  margin: 10px 0;
`;

const CollapsibleHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  cursor: pointer;
`;

const CollapsibleContent = styled.div`
  display: ${({ isCollapsed }) => (isCollapsed ? 'none' : 'flex')};
  flex-direction: column;
  margin-top: 10px;
`;

// Adjusted styles
const AppContainer = styled.div`
  display: flex;
  flex-direction: column;
  margin: 5px;
`;

const VideoContainer = styled.div`
  flex: 2;
  width: 60%; // Adjust according to your design needs
`;

const TranscriptContainer = styled.div`
  flex: 1;
  width: 20%; // Adjust according to your design needs
  height: 550px;
  overflow-y: scroll;
  border-left: 2px solid #ccc;
  padding: 10px;
`;

const TranscriptEntry = styled.p`
  cursor: pointer;
  &:hover {
    color: #007bff;
  }
  color: ${({ isActive }) => isActive ? '#007bff' : 'black'};
`;

const BeautifulText = styled.div`
  font-size: 16px;
  font-weight: 400;
  margin: 10px;
  padding: 10px;
  color: 'black';
  text-align: left;
  cursor: pointer;
`;

function App() {
  const [videoId, setVideoId] = useState("");
  const [videoUrl, setVideoUrl] = useState("");
  const [loading, setLoading] = useState(true);

  const [player, setPlayer] = useState(null);
  const [startTime, setStartTime] = useState(0);
  const [endTime, setEndTime] = useState(0); // Initialize with 0
  const [activeIndex, setActiveIndex] = useState(0);
  
  const [query, setQuery] = useState("");
  const [transcript, setTranscript] = useState([]);
  const [meta, setMeta] = useState("");
  const [selectedOption, setSelectedOption] = useState('transcript');

  const [isCollapsed, setIsCollapsed] = useState(true);

  const toggleCollapse = () => {
    setIsCollapsed(!isCollapsed);
  };


  useEffect(() => {
    const interval = setInterval(() => {
      if (player) {
        const currentTime = player.getCurrentTime();
        if (currentTime >= endTime) {
          //player.pauseVideo();
          moveToNextSegment();
        }
      }
    }, 100); // Check every second
    return () => clearInterval(interval);
  }, [player, endTime, activeIndex]);


  const moveToNextSegment = () => {
    const nextIndex = activeIndex + 1;
    if (nextIndex < transcript.length) {
      const nextSegment = transcript[nextIndex];
      handleTranscriptClick(nextSegment.start, nextSegment.end, nextIndex);
    }
  };

  const onReady = (event) => {
    setPlayer(event.target);
  };

  const handleTranscriptClick = (start, end, index) => {
    const newStartSeconds = parseFloat(start);
    const newEndSeconds = parseFloat(end);
    setEndTime(newEndSeconds);
    setActiveIndex(index);
    player.seekTo(newStartSeconds);
    //player.playVideo();
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    if (name === "ytlink") {
      setVideoUrl(value);
    } else if (name === "search") {
      setQuery(value);
    }
  };

  function getActiveIndex(timeInSeconds, transcript) {
    // Convert each time string to seconds and compare
    const index = transcript.findIndex(segment => {
        const startSeconds = parseFloat(segment.start)
        const endSeconds = parseFloat(segment.end)
        return timeInSeconds >= startSeconds && timeInSeconds <= endSeconds;
    });
    return index; // This will return -1 if no segment is active at the given time
  };


  async function fetchVideoAPI(myString) {
    const url = 'http://ec2-3-138-170-173.us-east-2.compute.amazonaws.com/add';
    const data = { link: myString };

    try {
      const response = await axios.post(url, data);
      console.log('Success:', response.data);
      return response.data; // You can return the data to use elsewhere
    } catch (error) {
      console.error('Error:', error);
    }
  }

  async function searchAPI() {
    const url = 'http://ec2-3-138-170-173.us-east-2.compute.amazonaws.com/search';
    const data = { query: query, table_cls: selectedOption, n: 5 };

    try {
      const response = await axios.post(url, data);
      console.log('Success:', response.data);
      return response.data; // You can return the data to use elsewhere
    } catch (error) {
      console.error('Error:', error);
    }
  }

  async function deleteAPI() {
    const url = 'http://ec2-3-138-170-173.us-east-2.compute.amazonaws.com/delete';
    const data = {};

    try {
      const response = await axios.post(url, data);
      console.log('Success:', response.data);
      return response.data; // You can return the data to use elsewhere
    } catch (error) {
      console.error('Error:', error);
    }
  }

  function getYouTubeVideoID(url) {
    const pattern = /(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?watch\?(?:\S*?&?v\=))|youtu\.be\/)([a-zA-Z0-9_-]{11})/;

    const match = url.match(pattern);
    return match ? match[1] : null;
  };

  const handleFetchYT = async () => {
    console.log("Fetching Video...")

    const newvideoId = getYouTubeVideoID(videoUrl);
    const response = await fetchVideoAPI(videoUrl)
    const obj = JSON.parse(response['transcript']);
    setTranscript(obj);
    setVideoId(newvideoId);
    setLoading(false)
  };

  const updateFetchYT = async (url=null) => {
    console.log("Fetching Video...", url)
    var response = {};
    var newVideoId = null;
    if (url) {
      newVideoId = getYouTubeVideoID(url);
      response = await fetchVideoAPI(url)
    } else {
      throw 'Invalid source url retrieved!';
    }
    const obj = JSON.parse(response['transcript']);
    setTranscript(obj);
    setVideoId(newVideoId);
    setLoading(false)
  };

  
  const handleSearch = async () => {
    console.log("Searching...");
    const response = await searchAPI(); 
    await updateFetchYT(response['source']);
    const newStartSeconds = parseFloat(response['start_time']);
    const newEndSeconds = parseFloat(response['end_time']);
    console.log("Seeking to, start: ", newStartSeconds, " end: ", newEndSeconds);
    player.seekTo(newStartSeconds);
  
    setStartTime(newStartSeconds);
    setEndTime(newEndSeconds);
    setMeta(response['meta']);
    
    if(response['source'] !== videoUrl) {
      setVideoUrl(response['source'])
    }
  }

  // Handle change event of the select dropdown
  const handleSelectChange = (event) => {
    setSelectedOption(event.target.value);
  };

  const handleDelete = async () => {
    const response = await deleteAPI();
    console.log(response);
  }

  const opts = {
    height: '550',
    width: '100%',
    playerVars: {
      autoplay: 1,
      start: startTime,
    },
  };


  return (
    <AppContainer>
      {loading ? (
        <div></div>
      ) : (
        <InputContainer>
        <input type="text" id="search" name="search" value={query} onChange={handleInputChange} />
        <select value={selectedOption} onChange={handleSelectChange}>
        <option value="transcript">transcript</option>
        <option value="frames">frames</option>
      </select>
        <button onClick={handleSearch}>Search Video</button>
      </InputContainer>
      )}
      {loading ? (
        <CollapsibleContainer>
          <CollapsibleHeader onClick={toggleCollapse}>
            <span>Input Section</span>
            <button>{isCollapsed ? 'Expand' : 'Collapse'}</button>
          </CollapsibleHeader>
          <CollapsibleContent isCollapsed={isCollapsed}>
          <InputContainer>
              <input type="text" id="ytlink" name="ytlink" value={videoUrl} onChange={handleInputChange} />
              <button onClick={handleFetchYT}>Fetch Video</button>
            </InputContainer>
          </CollapsibleContent>
        </CollapsibleContainer>
      ) : (
        <div>
          <CollapsibleContainer>
            <CollapsibleHeader onClick={toggleCollapse}>
              <span>Input Section</span>
              <button>{isCollapsed ? 'Expand' : 'Collapse'}</button>
            </CollapsibleHeader>
            <CollapsibleContent isCollapsed={isCollapsed}>
            <InputContainer>
                <input type="text" id="ytlink" name="ytlink" value={videoUrl} onChange={handleInputChange} />
                <button onClick={handleFetchYT}>Fetch Video</button>
              </InputContainer>
            </CollapsibleContent>
          </CollapsibleContainer>
        <div style={{ display: 'flex', width: '100%' }}>
          <VideoContainer>
            <YouTube videoId={videoId} opts={opts} onReady={onReady} />
          </VideoContainer>
          <TranscriptContainer>
            {transcript.map((item, index) => (
              <TranscriptEntry key={index} isActive={index === activeIndex} onClick={() => handleTranscriptClick(item.start, item.end, index)}>
                {item.text}
              </TranscriptEntry>
            ))}
          </TranscriptContainer>
        </div>
        <BeautifulText isActive={true}> {meta} </BeautifulText>
        
        <ButtonContainer>
          <button onClick={handleDelete}>Delete</button>
        </ButtonContainer>
        </div>
      )}
    </AppContainer>
  );
  
  
}

export default App;
