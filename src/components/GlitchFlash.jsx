import { useEffect, useState } from 'react';
import './GlitchFlash.css';

const GlitchFlash = () => {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const flashInterval = setInterval(() => {
      setIsVisible(true);
      
      // Hide after fade-in completes (200ms)
      setTimeout(() => {
        setIsVisible(false);
      }, 200);
    }, 15000); // Flash every 15 seconds

    return () => clearInterval(flashInterval);
  }, []);

  return (
    <div className={`glitch-flash ${isVisible ? 'visible' : ''}`}>
      <img src="/scarygirl.png" alt="" aria-hidden="true" />
    </div>
  );
};

export default GlitchFlash;
