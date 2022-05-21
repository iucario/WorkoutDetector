import { useState, useEffect, useRef, useCallback, Fragment } from "react";
import "./App.css";
import { io } from "socket.io-client";
import Webcam from "react-webcam";
import React from "react";

const socket = new WebSocket("ws://localhost:8000/ws") as WebSocket;

socket.onmessage = (msg) => {
  console.log(msg.data);
  showResult(msg.data);
};

const showResult = (data: string) => {
  // Display confidene
  const result = JSON.parse(data);
  let div = document.getElementById("result") as HTMLDivElement;
  div.innerHTML = "";
  for (const [key, value] of Object.entries(result)) {
    div.innerHTML += `${key}: ${value}<br>`;
  }

};

async function fetchPrediction(data: FormData | null) {
  const response = await fetch("http://localhost:8000/video", {
    method: "POST",
    body: data,
  });
  return response.json();
}

const WebcamStreamCapture = () => {
  const webcamRef = useRef(null) as React.MutableRefObject<any>;
  const mediaRecorderRef = useRef(null) as React.MutableRefObject<any>;
  const [capturing, setCapturing] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [intervalid, setIntervalid] = useState(null);
  const [recordedChunks, setRecordedChunks] = useState([]);

  const handleStartCaptureClick = useCallback(() => {
    setCapturing(true);
    mediaRecorderRef.current = new MediaRecorder(webcamRef.current.stream, {
      mimeType: "video/webm",
    });
    mediaRecorderRef.current.addEventListener(
      "dataavailable",
      handleDataAvailable
    );
    mediaRecorderRef.current.start();
  }, [webcamRef, setCapturing, mediaRecorderRef]);

  const handleDataAvailable = useCallback(
    ({ data }: any) => {
      if (data.size > 0) {
        setRecordedChunks((prev) => prev.concat(data));
      }
    },
    [setRecordedChunks]
  );

  const handleStopCaptureClick = useCallback(() => {
    mediaRecorderRef.current.stop();
    setCapturing(false);
  }, [mediaRecorderRef, webcamRef, setCapturing]);

  const handleDownload = useCallback(() => {
    if (recordedChunks.length) {
      const blob = new Blob(recordedChunks, {
        type: "video/webm",
      });
      const url = URL.createObjectURL(blob);
      let a = document.createElement("a");
      document.body.appendChild(a);
      // a.style = "display: none";
      a.href = url;
      a.download = "react-webcam-stream-capture.webm";
      a.click();
      window.URL.revokeObjectURL(url);
      setRecordedChunks([]);
    }
  }, [recordedChunks]);

  const sendVideo = () => {
    if (recordedChunks.length) {
      const blob = new Blob(recordedChunks, {
        type: "video/webm",
      });
      const formData = new FormData();
      formData.append("video", blob);
      fetchPrediction(formData)
        .then((res) => {
          console.log(res);
          console.log("Prediction success");
        })
        .catch((error) => {
          console.log(error);
          console.log("Failed to fetch");
        });
    }
  };
  const sendImage = () => {
    socket.send(webcamRef.current.getScreenshot());
  };
  const handleStartStream = () => {
    setStreaming(true);
    setIntervalid(window.setInterval(sendImage, 200) as any);
  };

  const handleStopStream = () => {
    setStreaming(false);
    if (intervalid) {
      window.clearInterval(intervalid);
    }
  };
  const handleStream = () => {
    if (streaming) {
      const image = webcamRef.current.getScreenshot();
      fetch("http://localhost:8000/image", {
        headers: {
          "Content-Type": "application/json",
        },
        method: "POST",
        body: JSON.stringify({ image: image }),
      })
        .then((response) => response.json())
        .then((result) => {
          console.log("Success:", result);
        })
        .catch((error) => {
          console.error("Error:", error);
        });
    }
  };

  useEffect(() => {
    handleStream();
  }, [recordedChunks]);

  return (
    <Fragment>
      <Webcam audio={false} ref={webcamRef} />
      {streaming ? (
        <button onClick={handleStopStream}>Stop Streaming</button>
      ) : (
        <button onClick={handleStartStream}>Start Streaming</button>
      )}
      {capturing ? (
        <button onClick={handleStopCaptureClick}>Stop Capture</button>
      ) : (
        <button onClick={handleStartCaptureClick}>Start Capture</button>
      )}
      {recordedChunks.length > 0 && (
        <button onClick={sendVideo}>Send video</button>
      )}
    </Fragment>
  );
};

const App = () => {
  return (
    <div>
      <WebcamStreamCapture />
      <div id="result">

      </div>
    </div>
  );
};

export default App;
