import { useState, useEffect, useRef, useCallback, Fragment } from "react";
import "./App.css";
import Webcam from "react-webcam";
import React from "react";
import { v4 as uuid } from "uuid";
import {
  Button,
  Grid,
  ButtonGroup,
  Box,
  TableHead,
  TableRow,
  TableCell,
  TableBody,
  Table,
  Typography,
  createTheme,
  useTheme,
  ThemeProvider,
} from "@mui/material";

import PlayCircleOutlineIcon from "@mui/icons-material/PlayCircleOutline";
import VideocamOutlinedIcon from "@mui/icons-material/VideocamOutlined";

const clientId = uuid().substring(0, 8);
console.log(clientId);
const socket = new WebSocket(
  `ws://localhost:8000/ws/${clientId}`
) as WebSocket;

const WebcamStreamCapture = ({ setResult }: any) => {
  const webcamRef = useRef(null) as any;
  const mediaRecorderRef = useRef(null) as any;
  const [capturing, setCapturing] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [intervalId, setIntervalId] = useState(null);
  const [recordedChunks, setRecordedChunks] = useState([]);

  const postVideo = (blob: Blob) => {
    const formData = new FormData();
    formData.append("video", blob);
    const url = "http://localhost:8000/video";
    fetch(url, {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((res) => {
        console.log(res);
        if (res.success) {
          console.log("Prediction success");
          setResult(res.data);
        }
      })
      .catch((error) => {
        console.log(error);
        console.log("Failed to fetch");
      });
  };

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

  useEffect(() => {
    if (recordedChunks.length) {
      const blob = new Blob(recordedChunks, {
        type: 'video/webm',
      })
      postVideo(blob);
      setRecordedChunks([]);
    }
  }, [recordedChunks]);
  const sendImage = () => {
    socket.send(webcamRef.current.getScreenshot());
  };
  const handleStartStream = () => {
    setStreaming(true);
    socket.send("start");
    setIntervalId(window.setInterval(sendImage, 200) as any);
  };

  const handleStopStream = () => {
    setStreaming(false);
    socket.send("stop");
    if (intervalId) {
      window.clearInterval(intervalId);
    }
  };

  const handleUploadVideoClick = () => {
    const uploadVideo = document.getElementById(
      "upload-video"
    ) as HTMLInputElement;
    uploadVideo.click();
  };

  const handleUploadVideoChange = (e: any) => {
    const file = e.target.files[0];
    if (file != null) {
      postVideo(file);
    }
  };
  const videoConstraints ={height: 720};
  return (
    <Fragment>
      <meta name="viewport" content="initial-scale=1, width=device-width" />
      <input
        id="upload-video"
        type="file"
        hidden
        onChange={handleUploadVideoChange}
      ></input>
      <Grid
        container
        spacing={2}
        columns={{ xs: 4, md: 12 }}
        direction="column"
        justifyContent="center"
        alignItems="center"
      >
        <Grid item xs={4} sm={6}>
          <Webcam audio={false} ref={webcamRef} videoConstraints={videoConstraints} />
        </Grid>
        <ButtonGroup variant="outlined">
          {streaming ? (
            <Button variant="outlined" onClick={handleStopStream}>
              <PlayCircleOutlineIcon />
              Stop Stream
            </Button>
          ) : (
            <Button variant="outlined" onClick={handleStartStream}>
              <PlayCircleOutlineIcon />
              Start Stream
            </Button>
          )}

          {capturing ? (
            <Button variant="outlined" onClick={handleStopCaptureClick}>
              <VideocamOutlinedIcon />
              Stop Capture
            </Button>
          ) : (
            <Button variant="outlined" onClick={handleStartCaptureClick}>
              <VideocamOutlinedIcon />
              Start Capture
            </Button>
          )}
          <Button variant="outlined" onClick={handleUploadVideoClick}>
            Upload Video
          </Button>
        </ButtonGroup>
      </Grid>
    </Fragment>
  );
};

const App = () => {
  let theme = useTheme() as any;
  const dummy = {
    mountain_climber: 0,
    lunge: 0,
    exercising_arm: 0,
    front_raise: 0,
    squat: 0,
    push_up: 0,
    jumping_jack: 0,
    pull_up: 0,
    situp: 0,
    bench_pressing: 0,
    battle_rope: 0,
  };
  const [result, setResult] = useState(dummy);

  socket.onmessage = (msg) => {
    console.log(msg.data);
    const json = JSON.parse(msg.data);
    if (json.success) {
      setResult(json.data);
    }
  };

  theme = createTheme({
    typography: {
      htmlFontSize: 15,
      h3: {
        fontSize: "1.5rem",
        "@media (min-width:600px)": {
          fontSize: "2rem",
        },
      },
      h4: {
        fontSize: "1.2rem",
        "@media (min-width:600px)": {
          fontSize: "1.5rem",
        },
      },
      body1: {
        fontSize: "1rem",
        "@media (min-width:600px)": {
          fontSize: "1.2rem",
        },
      },
      fontFamily: [
        "-apple-system",
        "BlinkMacSystemFont",
        "Roboto",
        '"Helvetica Neue"',
        '"Segoe UI"',
        "Arial",
        "sans-serif",
        '"Apple Color Emoji"',
        '"Segoe UI Emoji"',
        '"Segoe UI Symbol"',
      ].join(","),
    },
    palette: {
      primary: {
        light: "#757ce8",
        main: "#673ab7",
        dark: "#002884",
        contrastText: "#fff",
      },
      secondary: {
        light: "#ff7961",
        main: "#2979ff",
        dark: "#ba000d",
        contrastText: "#000",
      },
    },
    components: {
      MuiTypography: {
        defaultProps: {
          variantMapping: {
            h1: "h3",
            h2: "h3",
            h3: "h3",
            h4: "h4",
            h5: "h5",
            h6: "h6",
            subtitle1: "h2",
            subtitle2: "h2",
            body1: "span",
            body2: "span",
          },
        },
      },
    },
  });

  return (
    <>
      <ThemeProvider theme={theme}>
        <Box alignContent="center" margin={2}>
          <Typography variant="h3" component="h1">
            Workout Detector
          </Typography>
          <Typography variant="h4" component="h2">
            Click "start streaming" to show real time inference results. Click
            "start capture" to record a video and get results of that video.
          </Typography>
        </Box>
        <Grid
          container
          spacing={2}
          columns={{ xs: 4, md: 12 }}
          direction="row"
          justifyContent="center"
          alignItems="center"
        >
          <Grid item xs={4} md={7}>
            <WebcamStreamCapture setResult={setResult} />
          </Grid>
          <Grid item xs={4} md={3} alignItems="center" sx={{ maxWidth: 300 }}>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell align="center">Workout</TableCell>
                  <TableCell align="center">Confidence</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {Object.entries(result).map(([k, v]) => (
                  <TableRow key={k}>
                    <TableCell align="center">{k}</TableCell>
                    <TableCell align="center">{v.toFixed(3)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Grid>
        </Grid>
      </ThemeProvider>
    </>
  );
};

export default App;
