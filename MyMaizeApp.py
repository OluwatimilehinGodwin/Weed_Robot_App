import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tempfile
import os

st.set_page_config(page_title="Maize vs Weed Detection", layout="centered")

# Config

DEFAULT_MODEL_PATH = "weedmanbeta.pt"  # relative path, works if shipped alongside the app

# Class-specific box colors (BGR for OpenCV). Extend/edit as needed to match
# your model's actual class names (run model.names to check).
CLASS_COLORS = {
    "maize": (0, 255, 0),     # green
    "weed": (0, 0, 255),      # red
}
DEFAULT_COLOR = (255, 165, 0)  # orange, used for any class not in the map above

CONF_THRESHOLD = 0.25


# Model loading (cached + lazy: only runs once an image triggers it)

@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    """Load a YOLO model exactly once per unique path, cached across reruns."""
    from ultralytics import YOLO
    return YOLO(model_path)


def get_model_path() -> str | None:
    """
    Resolve the model path for this run:
    - if the user uploaded a .pt file, save it to a temp file and use that
    - otherwise fall back to DEFAULT_MODEL_PATH if it exists on disk
    """

    if os.path.exists(DEFAULT_MODEL_PATH):
        return DEFAULT_MODEL_PATH

    return None


# Drawing

def draw_detections(image_np: np.ndarray, result) -> tuple[np.ndarray, int]:
    """
    Draw bounding boxes + confidence labels onto a copy of image_np.
    Returns (annotated_image, num_detections).
    """
    annotated = image_np.copy()
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return annotated, 0

    names = result.names  # {class_id: class_name}

    for box in boxes:
        conf = float(box.conf[0])
        if conf < CONF_THRESHOLD:
            continue

        cls_id = int(box.cls[0])
        cls_name = names.get(cls_id, str(cls_id))
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        color = CLASS_COLORS.get(cls_name.lower(), DEFAULT_COLOR)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label = f"{cls_name} {conf:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        label_y1 = max(y1 - text_h - baseline - 4, 0)
        cv2.rectangle(
            annotated, (x1, label_y1), (x1 + text_w + 4, y1), color, -1
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 2, y1 - baseline - 2 if label_y1 > 0 else y1 + text_h + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return annotated, sum(1 for b in boxes if float(b.conf[0]) >= CONF_THRESHOLD)


# App

st.title("Maize vs Weed Detection")
st.caption("Upload an image or capture one from your camera to run detection.")

model_path = get_model_path()

if model_path is None:
    st.warning(
        "No model found. Upload a `.pt` YOLO model file in the sidebar to get started."
    )
    st.stop()

# Reset trigger: bump a counter to clear uploader/camera widget state
if "reset_counter" not in st.session_state:
    st.session_state["reset_counter"] = 0

col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        key=f"uploader_{st.session_state['reset_counter']}",
    )
with col2:
    cam_file = st.camera_input(
        "Or capture image with camera",
        key=f"camera_{st.session_state['reset_counter']}",
    )

img_file = uploaded_file if uploaded_file is not None else cam_file

if img_file is not None:
    image = Image.open(img_file).convert("RGB")
    image_np = np.array(image)

    # Model is initialized lazily here, on first image — cached after that,
    # so subsequent images/reruns reuse the same loaded model in memory.
    with st.spinner("Loading model..."):
        model = load_model(model_path)

    with st.spinner("Running inference..."):
        results = model.predict(image_np, conf=CONF_THRESHOLD, verbose=False)
        result = results[0]

    annotated, num_detections = draw_detections(image_np, result)

    if num_detections > 0:
        st.image(annotated, caption=f"{num_detections} detection(s)", use_container_width=True)

        st.subheader("Detections")
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < CONF_THRESHOLD:
                continue
            cls_name = result.names.get(int(box.cls[0]), "unknown")
            st.write(f"- **{cls_name}** — confidence: {conf:.2%}")
    else:
        st.image(image_np, caption="No detections found", use_container_width=True)
        st.info("No maize or weed detected in this image.")

    if st.button("🔄 Run again"):
        st.session_state["reset_counter"] += 1
        st.rerun()
