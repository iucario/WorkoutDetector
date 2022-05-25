import { useState, useEffect, useRef, useCallback, Fragment } from "react";
import "./App.css";
import Webcam from "react-webcam";
import React from "react";
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

const socket = new WebSocket("ws://"+ window.location.hostname + ":8000/ws") as WebSocket;

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
    sendVideo();
  }, [mediaRecorderRef, webcamRef, setCapturing]);

  const sendVideo = () => {
    if (recordedChunks.length) {
      const blob = new Blob(recordedChunks, {
        type: "video/webm",
      });
      const formData = new FormData();
      formData.append("video", blob);
      const url = "http://" + window.location.hostname + ":8000/video";
      fetch(url, {
        method: "POST",
        body: formData,
      })
        .then((response) => response.json())
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

  return (
    <Fragment>
      <meta name="viewport" content="initial-scale=1, width=device-width" />
      <Grid
        container
        spacing={2}
        columns={{ xs: 4, md: 12 }}
        direction="column"
        justifyContent="center"
        alignItems="center"
      >
        <Grid item xs={4} sm={6}>
          <Webcam audio={false} ref={webcamRef} />
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
    setResult(JSON.parse(msg.data));
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
            <WebcamStreamCapture />
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
