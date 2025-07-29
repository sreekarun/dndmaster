import React from 'react';
import './Home.css';
import DemonMouth from './DemonMouth';

export default function Home() {
  return (
    <div className="container">
      <h1>Dungeons & Dragons Master</h1>
        <h2>Welcome to the D&D Master! Your adventure begins here.</h2>
      <div className="demon-face">
        <div className="horns">
          <div className="horn left" />
          <div className="horn right" />
        </div>
        <div className="face">
          <div className="eyes">
            <div className="eye left" />
            <div className="eye right" />
          </div>
          <div className="nose" />
          <DemonMouth />
        </div>
      </div>
    </div>
  );
}
