import argparse
import asyncio
import json
import logging
import os
import platform
from contextlib import asynccontextmanager

import cv2
import numpy as np
from av import VideoFrame
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaPlayer, MediaRelay
from ultralytics import YOLO
import onnxruntime
import requests
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import gradio as gr
import uvicorn

# Define a color map for different classes
color_map = {
    0: (209, 54, 40),  # Red
    1: (37, 194, 45),  # Green
    2: (34, 92, 240),  # Blue
}

ROOT = os.path.dirname(__file__)

# File download URLs
file_urls = [
    'https://www.dropbox.com/scl/fi/5i223blrvkarlczd6v175/3002177.jpg?rlkey=mcfhy20cpu0hltqdcy7jtohin&st=noqsik1v&dl=1',
    'https://www.dropbox.com/scl/fi/7u4e56zdz5i5gsbneo9jv/Screenshot-2024-06-12-161636.png?rlkey=0tuqjo51ptop3aa4rzissvmio&st=2oh7ko6z&dl=1',
    'https://www.dropbox.com/scl/fi/jdeana2t9lzec0czwvafu/1116839_Lesson_Hand_1280x720.mp4?rlkey=qbahgmbk3jjm2g6t0obl53uw0&st=w2jlo6oc&dl=1'
]

# Download files
def download_file(url, save_name):
    if not os.path.exists(save_name):
        file = requests.get(url)
        open(save_name, 'wb').write(file.content)

for i, url in enumerate(file_urls):
    save_path = os.path.join(ROOT, f"image_{i}.jpg" if 'mp4' not in url else "video.mp4")
    download_file(url, save_path)

# Initialize YOLO model
model = YOLO("/Users/minhtien/Developer/Learning/Classroom-Behavior-Detector/best.pt")

# WebRTC setup
relay = None
webcam = None
pcs = set()

class YOLOVideoStreamTrack(VideoStreamTrack):
    """
    A video track that returns camera track with annotated detected objects.
    """
    def __init__(self, conf_thres=0.7, iou_thres=0.5):
        super().__init__()
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres

        video = cv2.VideoCapture(0)
        if not video.isOpened():
            logging.error("Failed to open webcam")
            raise RuntimeError("Could not open webcam")
        self.video = video

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        ret, frame = self.video.read()
        if not ret:
            logging.warning("Failed to read frame from webcam")
            return None
        frame = self.detect_classroom(frame)
        frame = VideoFrame.from_ndarray(frame, format="bgr24")
        frame.pts = pts
        frame.time_base = time_base
        return frame
    
    def detect_classroom(self, frame):
        """Use the classroom behavior detection model to detect student behaviors."""
        results = model.predict(source=frame, verbose=False)[0]
        
        for i, det in enumerate(results.boxes.xyxy):
            x1, y1, x2, y2 = map(int, det)
            cls = int(results.boxes.cls[i].item())
            label = results.names[cls]
            confidence = results.boxes.conf[i].item()
            color = color_map.get(cls, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        
        return frame

    def __del__(self):
        if hasattr(self, 'video') and self.video is not None:
            self.video.release()


def create_local_tracks():
    global relay, webcam
    options = {"framerate": "30", "video_size": "640x480"}
    if relay is None:
        try:
            if platform.system() == "Darwin":
                webcam = MediaPlayer(
                    "default:none", format="avfoundation", options=options
                )
            elif platform.system() == "Windows":
                webcam = MediaPlayer(
                    "video=Integrated Camera", format="dshow", options=options
                )
            else:
                webcam = MediaPlayer("/dev/video0", format="v4l2", options=options)
            relay = MediaRelay()
        except Exception as e:
            logging.error(f"Failed to initialize webcam: {e}")
            raise
    return relay.subscribe(webcam.video)


# Gradio interfaces
def show_preds_image(image_path):
    if isinstance(image_path, dict):
        image_path = image_path['name']
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    outputs = model.predict(source=image_path)
    results = outputs[0]
    for i, det in enumerate(results.boxes.xyxy):
        x1, y1, x2, y2 = map(int, det)
        cls = int(results.boxes.cls[i].item())
        label = results.names[cls]
        confidence = results.boxes.conf[i].item()
        color = color_map.get(cls, (255, 255, 255))
        cv2.rectangle(image, (x1, y1), (x2, y2), color=color, thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(image, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def show_preds_video(video_path):
    if isinstance(video_path, dict):
        video_path = video_path['name']
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    output_video_path = os.path.join(ROOT, 'output_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            outputs = model.predict(source=frame)
            results = outputs[0]
            for i, det in enumerate(results.boxes.xyxy):
                x1, y1, x2, y2 = map(int, det)
                cls = int(results.boxes.cls[i].item())
                label = results.names[cls]
                confidence = results.boxes.conf[i].item()
                color = color_map.get(cls, (255, 255, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=2, lineType=cv2.LINE_AA)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
            out.write(frame)
        else:
            break

    cap.release()
    out.release()
    return output_video_path

# Gradio inference interfaces
path = [[os.path.join(ROOT, 'image_0.jpg')], [os.path.join(ROOT, 'image_1.jpg')]]
video_path = [[os.path.join(ROOT, 'video.mp4')]]

inputs_image = [
    gr.components.Image(type="filepath", label="Input Image"),
]
outputs_image = [
    gr.components.Image(type="numpy", label="Output Image"),
]
interface_image = gr.Interface(
    fn=show_preds_image,
    inputs=inputs_image,
    outputs=outputs_image,
    title="Student-Behavior-Detector-in-Classroom",
    examples=path,
    cache_examples=False,
)

inputs_video = [
    gr.components.Video(label="Input Video"),
]
outputs_video = [
    gr.components.Video(label="Output Video"),
]
interface_video = gr.Interface(
    fn=show_preds_video,
    inputs=inputs_video,
    outputs=outputs_video,
    title="Student-Behavior-Detector-in-Classroom",
    examples=video_path,
    cache_examples=False,
)

# Custom Gradio theme to avoid font errors
custom_theme = gr.themes.Default(
    font=[gr.themes.GoogleFont("Roboto Mono"), "monospace", "sans-serif"],
    text_size=gr.themes.sizes.text_md
)

# FastAPI setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO)
    yield
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

app = FastAPI(lifespan=lifespan)

# app.mount("/webrtc-static", StaticFiles(directory=ROOT, html=True), name="webrtc-static")

# WebRTC routes
@app.get("/webrtc")
async def get_webrtc():
    with open(os.path.join(ROOT, "index.html"), "r") as f:
        # html_content = f.read().replace('src="client.js"', 'src="/webrtc-static/client.js"')
        html_content = f.read()
    return HTMLResponse(content=html_content)

# @app.get("/webrtc-static/client.js")
@app.get("/client.js")
async def get_javascript():
    return FileResponse(os.path.join(ROOT, "client.js"))

@app.post("/offer")
async def offer(request: Request):
    try:
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        pc = RTCPeerConnection()
        pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logging.info(f"Connection state is {pc.connectionState}")
            if pc.connectionState == "failed":
                await pc.close()
                pcs.discard(pc)

        video = create_local_tracks()
        if video:
            pc.addTrack(YOLOVideoStreamTrack())
        else:
            logging.error("Failed to create local video track")
            raise ValueError("No video track available")

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        }
    except Exception as e:
        logging.error(f"Error in /offer endpoint: {e}")
        return {"error": str(e)}, 500

# Mount Gradio inference app
inference_interface = gr.TabbedInterface(
    [interface_image, interface_video],
    tab_names=['Image inference', 'Video inference'],
    theme=custom_theme
)
app = gr.mount_gradio_app(app, inference_interface, path="/")

# Mount Gradio WebRTC app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebRTC and Gradio integrated FastAPI app")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )

    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)