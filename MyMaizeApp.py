import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os
import threading
import time
import queue
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

st.set_page_config(
    page_title="Maize vs Weed Detection",
    layout="wide",
    page_icon="🌽",
)

# Config
DEFAULT_MODEL_PATH = r"C:\Users\USER\Desktop\python\weedman.pt"

CLASS_COLORS = {
    "maize": (0, 200, 80),   # green
    "weed":  (220, 50, 50),  # red
}
DEFAULT_COLOR = (255, 140, 0)  # orange

CONF_THRESHOLD = 0.25

# WebRTC ICE servers (STUN only; for LAN/Pi deployment this is usually enough)
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


# Model loading
@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    from ultralytics import YOLO
    return YOLO(model_path)


def get_model_path() -> str | None:
    if os.path.exists(DEFAULT_MODEL_PATH):
        return DEFAULT_MODEL_PATH
    return None



# Drawing helper
def draw_detections(image_np: np.ndarray, result) -> tuple[np.ndarray, list[dict]]:
    annotated = image_np.copy()
    boxes = result.boxes
    detections = []

    if boxes is None or len(boxes) == 0:
        return annotated, detections

    names = result.names

    for box in boxes:
        conf = float(box.conf[0])
        if conf < CONF_THRESHOLD:
            continue

        cls_id  = int(box.cls[0])
        cls_name = names.get(cls_id, str(cls_id))
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        bgr = CLASS_COLORS.get(cls_name.lower(), DEFAULT_COLOR)
        # OpenCV uses BGR
        bgr_cv = (bgr[2], bgr[1], bgr[0])

        cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr_cv, 2)

        label = f"{cls_name} {conf:.0%}"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ly1 = max(y1 - th - bl - 4, 0)
        cv2.rectangle(annotated, (x1, ly1), (x1 + tw + 4, y1), bgr_cv, -1)
        cv2.putText(
            annotated, label,
            (x1 + 2, y1 - bl - 2 if ly1 > 0 else y1 + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )
        detections.append({"class": cls_name, "conf": conf})

    return annotated, detections



# Live stream video processor
class YOLOVideoProcessor:
    """
    Called once per video frame by streamlit-webrtc.
    Runs YOLO inference and overlays detections in real time,
    mimicking what runs on a Raspberry Pi camera feed.
    """

    def __init__(self, model):
        self.model = model
        self.result_queue: queue.SimpleQueue = queue.SimpleQueue()
        self._lock = threading.Lock()
        self.fps_start = time.time()
        self.frame_count = 0
        self.fps = 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Run inference
        results = self.model.predict(img, conf=CONF_THRESHOLD, verbose=False)
        result  = results[0]

        annotated, detections = draw_detections(img, result)

        # FPS overlay (like a Pi camera preview)
        self.frame_count += 1
        elapsed = time.time() - self.fps_start
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_start = time.time()

        cv2.putText(
            annotated,
            f"FPS: {self.fps:.1f}",
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (255, 255, 0), 2, cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"Detections: {len(detections)}",
            (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (255, 255, 0), 2, cv2.LINE_AA,
        )

        # Push latest detections for the sidebar panel (non-blocking)
        if not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                pass
        self.result_queue.put(detections)

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")



# UI styles
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #0f1a0f; }
    [data-testid="stSidebar"]          { background: #121f12; border-right: 1px solid #2a4a2a; }
    h1  { color: #7ec850 !important; letter-spacing: -0.5px; }
    h2, h3 { color: #a8d87e !important; }
    .stCaption { color: #6b8f5e !important; }

    .badge-maize {
        background: #1a4a1a; color: #7ec850;
        padding: 3px 10px; border-radius: 12px;
        font-size: 0.85rem; font-weight: 600;
        border: 1px solid #3a7a3a; display: inline-block;
    }
    .badge-weed {
        background: #4a1a1a; color: #e85555;
        padding: 3px 10px; border-radius: 12px;
        font-size: 0.85rem; font-weight: 600;
        border: 1px solid #8a3a3a; display: inline-block;
    }
    .badge-other {
        background: #3a2a0a; color: #ffaa44;
        padding: 3px 10px; border-radius: 12px;
        font-size: 0.85rem; font-weight: 600;
        border: 1px solid #7a5a1a; display: inline-block;
    }
    .stat-card {
        background: #1a2e1a; border: 1px solid #2a4a2a;
        border-radius: 10px; padding: 14px 18px; margin: 6px 0;
        text-align: center;
    }
    .stat-number { font-size: 2rem; font-weight: 700; color: #7ec850; }
    .stat-label  { font-size: 0.78rem; color: #6b8f5e; text-transform: uppercase;
                   letter-spacing: 0.08em; }
    .mode-chip {
        display: inline-block; padding: 4px 12px; border-radius: 20px;
        font-size: 0.8rem; font-weight: 600; margin-bottom: 10px;
    }
    .live-chip  { background: #ff3b3b22; color: #ff6b6b; border: 1px solid #ff3b3b55; }
    .still-chip { background: #3b7fff22; color: #7baaff; border: 1px solid #3b7fff55; }
</style>
""", unsafe_allow_html=True)



# Sidebar
with st.sidebar:
    st.markdown("## 🌽 WeedMan")
    st.markdown("---")

    mode = st.radio(
        "Detection Mode",
        options=["📷 Still Image", "🎥 Live Stream"],
        index=0,
        help="Live Stream mimics Raspberry Pi camera deployment.",
    )
    st.markdown("---")

    st.markdown("**Confidence threshold**")
    conf_thresh = st.slider("", 0.10, 0.90, CONF_THRESHOLD, 0.05,
                             label_visibility="collapsed")
    CONF_THRESHOLD = conf_thresh

    st.markdown("---")
    st.markdown("**Model**")
    model_path = get_model_path()
    if model_path:
        st.success(f"✅ `{os.path.basename(model_path)}`")
    else:
        st.error("No model found on disk.")
        st.info("Place `weedmanbeta.pt` alongside `app.py` and restart.")



# Header
st.title("Maize vs Weed Detection")
st.caption("Upload a still image — or stream live from your camera just like on a Raspberry Pi.")

if model_path is None:
    st.error("⚠️ No model loaded. See sidebar for instructions.")
    st.stop()

model = load_model(model_path)


#  MODE 1 — Still Image
if mode == "📷 Still Image":
    st.markdown('<span class="mode-chip still-chip">📷 Still Image Mode</span>',
                unsafe_allow_html=True)

    if "reset_counter" not in st.session_state:
        st.session_state["reset_counter"] = 0

    col_up, col_cam = st.columns(2)
    with col_up:
        uploaded_file = st.file_uploader(
            "Upload an image", type=["jpg", "jpeg", "png"],
            key=f"uploader_{st.session_state['reset_counter']}",
        )
    with col_cam:
        cam_file = st.camera_input(
            "Or capture from camera",
            key=f"camera_{st.session_state['reset_counter']}",
        )

    img_file = uploaded_file if uploaded_file is not None else cam_file

    if img_file is not None:
        image    = Image.open(img_file).convert("RGB")
        image_np = np.array(image)

        with st.spinner("Running inference…"):
            results   = model.predict(image_np, conf=CONF_THRESHOLD, verbose=False)
            result    = results[0]

        annotated, detections = draw_detections(image_np, result)

        col_img, col_info = st.columns([2, 1])
        with col_img:
            if detections:
                st.image(annotated,
                         caption=f"{len(detections)} detection(s)",
                         use_container_width=True)
            else:
                st.image(image_np,
                         caption="No detections found",
                         use_container_width=True)
                st.info("No maize or weed detected.")

        with col_info:
            maize_n = sum(1 for d in detections if d["class"].lower() == "maize")
            weed_n  = sum(1 for d in detections if d["class"].lower() == "weed")

            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{len(detections)}</div>
                <div class="stat-label">Total detections</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" style="color:#7ec850">{maize_n}</div>
                <div class="stat-label">Maize</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" style="color:#e85555">{weed_n}</div>
                <div class="stat-label">Weed</div>
            </div>
            """, unsafe_allow_html=True)

            if detections:
                st.markdown("**Detections**")
                for d in detections:
                    cls = d["class"].lower()
                    badge = "badge-maize" if cls == "maize" else \
                            "badge-weed"  if cls == "weed"  else "badge-other"
                    st.markdown(
                        f'<span class="{badge}">{d["class"]}</span> '
                        f'<span style="color:#6b8f5e">{d["conf"]:.1%}</span>',
                        unsafe_allow_html=True,
                    )

        if st.button("🔄 Run again"):
            st.session_state["reset_counter"] += 1
            st.rerun()


#  MODE 2 — Live Stream  (mimics Raspberry Pi deployment)
else:
    st.markdown('<span class="mode-chip live-chip">🔴 Live Stream Mode</span>',
                unsafe_allow_html=True)

    st.info(
        "**How it works:** Each video frame is passed through the YOLO model in real time, "
        "exactly as it would run on a Raspberry Pi with a Pi Camera. "
        "Click **START** below to begin streaming."
    )

    col_stream, col_stats = st.columns([3, 1])

    with col_stream:
        ctx = webrtc_streamer(
            key="yolo-live",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIG,
            video_processor_factory=lambda: YOLOVideoProcessor(model),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    with col_stats:
        st.markdown("### Live Stats")
        detection_placeholder = st.empty()
        maize_placeholder     = st.empty()
        weed_placeholder      = st.empty()
        log_placeholder       = st.empty()

        # Poll the result queue while stream is active
        if ctx.video_processor:
            st.markdown("---")
            st.markdown("**Last frame detections**")
            log_box = st.empty()

            while ctx.state.playing:
                try:
                    detections = ctx.video_processor.result_queue.get(timeout=0.5)
                except queue.Empty:
                    detections = []

                maize_n = sum(1 for d in detections if d["class"].lower() == "maize")
                weed_n  = sum(1 for d in detections if d["class"].lower() == "weed")

                detection_placeholder.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{len(detections)}</div>
                    <div class="stat-label">Detections (frame)</div>
                </div>""", unsafe_allow_html=True)

                maize_placeholder.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number" style="color:#7ec850">{maize_n}</div>
                    <div class="stat-label">Maize</div>
                </div>""", unsafe_allow_html=True)

                weed_placeholder.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number" style="color:#e85555">{weed_n}</div>
                    <div class="stat-label">Weed</div>
                </div>""", unsafe_allow_html=True)

                if detections:
                    lines = []
                    for d in detections:
                        cls   = d["class"].lower()
                        badge = "badge-maize" if cls == "maize" else \
                                "badge-weed"  if cls == "weed"  else "badge-other"
                        lines.append(
                            f'<span class="{badge}">{d["class"]}</span> '
                            f'<span style="color:#6b8f5e">{d["conf"]:.1%}</span>'
                        )
                    log_box.markdown("<br>".join(lines), unsafe_allow_html=True)
                else:
                    log_box.markdown(
                        '<span style="color:#6b8f5e">Nothing detected</span>',
                        unsafe_allow_html=True,
                    )

                time.sleep(0.1)

    st.markdown("---")
    with st.expander("🔧 Raspberry Pi deployment notes"):
        st.markdown("""
        This live stream mode closely simulates how the model runs on a Pi:

        - **Frame loop**: Every frame from the camera goes through `model.predict()` —
          same call used in Pi Camera scripts.
        - **HTTPS**: `streamlit-webrtc` requires HTTPS for camera access in browsers.
          On a Pi, use `ngrok` or a reverse proxy with a self-signed cert.
        - **Performance**: On Pi 4/5 with a quantized YOLOv8n model, expect ~3–8 FPS.
          The FPS counter on the stream overlay lets you monitor this directly.
        - **Pi Camera swap**: Replace the WebRTC source with `picamera2` + OpenCV in
          your Pi deployment script. The `draw_detections()` function is reusable as-is.
        """)
