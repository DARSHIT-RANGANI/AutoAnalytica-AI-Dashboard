import React from "react";
import Upload from "./components/Upload";
import bgImage from "./assets/bg.jpg";

function App() {
  return (
    <div
      className="min-h-screen bg-cover bg-center relative"
      style={{ backgroundImage: `url(${bgImage})` }}
    >
      {/* Dark Overlay */}
      <div className="absolute inset-0 bg-black/70 backdrop-blur-sm"></div>

      {/* Main Content */}
      <div className="relative z-10 text-white p-10">
        <h1 className="text-4xl font-bold text-center mb-8">
          🚀 AutoAnalytica AI Dashboard
        </h1>

        <Upload />
      </div>
    </div>
  );
}

export default App;