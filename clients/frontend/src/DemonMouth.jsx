import React, { useRef } from 'react';
import screamAudio from './assets/scream.mp3';

export default function DemonMouth() {
  const audioRef = useRef(null);

  const handleMouseEnter = () => {
    if (audioRef.current) {
      audioRef.current.currentTime = 0;
      audioRef.current.play();
    }
  };

  return (
    <>
      <div className="mouth" onMouseEnter={handleMouseEnter} />
      <audio ref={audioRef} src={screamAudio} preload="auto" />
    </>
  );
}
