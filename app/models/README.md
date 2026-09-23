# app/models/

Place `hand_landmarker.task` here to enable real hand tracking in
`practice_session.py` (instead of the default fixed-center ROI box).

Download it from Google's official MediaPipe model host:

```
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

Save it as exactly:

```
app/models/hand_landmarker.task
```

See the main `README.md`, section "Enabling real hand tracking", for
full details and the fixed-box fallback behavior if this file is absent.
