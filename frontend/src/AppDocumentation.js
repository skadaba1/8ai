import React, { useState } from 'react';
import './AppDocumentation.css';

const AppDocumentation = () => {
  const [activeIndex, setActiveIndex] = useState(null);
  const [isCollapsed, setIsCollapsed] = useState(false);

  const toggleSection = (index) => {
    setActiveIndex(activeIndex === index ? null : index);
  };

  const toggleCollapse = () => {
    setIsCollapsed(!isCollapsed);
  };

  const toggle = (index) => {
    setActiveIndex(activeIndex === index ? null : index);
  };

  return (
    <div className="container">
      <button onClick={toggleCollapse} className="collapse-toggle">
        {isCollapsed ? 'Show Documentation' : 'Hide Documentation'}
      </button>
      {!isCollapsed && (
      <div>
      <h1>App Usage Documentation</h1>
      <div className="accordion">
        <div className="accordion-item">
          <button className="accordion-title" onClick={() => toggle(0)}>
            Uploading and Managing Videos
          </button>
          <div className={`accordion-content ${activeIndex === 0 ? 'show' : ''}`}>
            <p>Expand the video input field to upload a new YouTube video by entering its URL. Once uploaded, the video can be managed within the app's interface. Videos longer than 20 minutes will have transcripts broken into chunks of 30 seconds. Otherwise, they will be broken based on natural breaks and speaker changes. </p>
          </div>
        </div>
        <div className="accordion-item">
          <button className="accordion-title" onClick={() => toggle(1)}>
            Searching in Videos
          </button>
          <div className={`accordion-content ${activeIndex === 1 ? 'show' : ''}`}>
            <p>Toggle between frames and transcript options to search through images or text across all videos you have uploaded. Utilize the search bar to find specific content within the videos.</p>
          </div>
        </div>
        <div className="accordion-item">
          <button className="accordion-title" onClick={() => toggle(2)}>
            Navigating and Viewing Transcripts
          </button>
          <div className={`accordion-content ${activeIndex === 2 ? 'show' : ''}`}>
            <p>Click any part of the transcript beside the video player to jump to that specific part of the video. The transcript will also highlight to show the current playback position.</p>
          </div>
        </div>
        <div className="accordion-item">
          <button className="accordion-title" onClick={() => toggle(3)}>
            Browsing Search Results
          </button>
          <div className={`accordion-content ${activeIndex === 3 ? 'show' : ''}`}>
            <p>Use the "Next" and "Prev" buttons to navigate through search results. These buttons become functional after performing a search, allowing you to browse through the search results.</p>
          </div>
        </div>
        <div className="accordion-item">
          <button className="accordion-title" onClick={() => toggle(4)}>
            Managing Video Cache
          </button>
          <div className={`accordion-content ${activeIndex === 4 ? 'show' : ''}`}>
            <p>Click the "Delete" button to clear the video cache when you are finished using the application for the day. This ensures that storage does not become overly full.</p>
          </div>
        </div>
      </div>
      </div>
      )}
    </div>
  );
};

export default AppDocumentation;
