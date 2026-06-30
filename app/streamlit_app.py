from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mps_traffic_density.density import density_threshold_description
from mps_traffic_density.detector import load_yolo_model, predict_image
from mps_traffic_density.visualization import draw_prediction


st.set_page_config(page_title="Traffic Signal Vehicle Detection", layout="wide")

FINAL_YOLO_WEIGHTS = Path("models/bmd45_yolo11s_gpu_50epochs_best.pt")
LOCAL_YOLO_WEIGHTS = Path("runs/detect/runs/detect/bmd45_yolo11s_gpu_50epochs/weights/best.pt")
RTDETR_WEIGHTS = Path("runs/detect/runs/detect/bmd45_rtdetr_l_gpu_20epochs_300/weights/best.pt")


@st.cache_resource(show_spinner=False)
def cached_model(weights: str):
    return load_yolo_model(weights)


def first_available_weight() -> str:
    for path in (FINAL_YOLO_WEIGHTS, LOCAL_YOLO_WEIGHTS):
        if path.exists():
            return str(path)
    return "yolo11n.pt"


def model_choices() -> dict[str, str]:
    choices = {
        "YOLO11s final GPU model": first_available_weight(),
        "YOLO11s COCO traffic-general baseline": "yolo11s.pt",
        "YOLO11n public baseline": "yolo11n.pt",
    }
    if RTDETR_WEIGHTS.exists():
        choices["RT-DETR-L transformer comparison"] = str(RTDETR_WEIGHTS)
    choices["Custom path"] = ""
    return choices


def main() -> None:
    st.title("Traffic Signal Vehicle Detection")
    st.caption("Upload road or traffic-signal images to detect and label vehicles, count classes, and estimate traffic density.")

    choices = model_choices()
    with st.sidebar:
        st.header("Model")
        selected_model = st.selectbox("Detector", list(choices.keys()))
        st.caption(
            "Use the final GPU model for BMD-45/demo images. "
            "For Google/phone highway photos with many small vehicles, try the COCO traffic-general baseline."
        )
        custom_weights = ""
        if selected_model == "Custom path":
            custom_weights = st.text_input("Weights path", value=first_available_weight())
        weights = custom_weights or choices[selected_model]
        st.text_input("Active weights", value=weights, disabled=True)

        confidence = st.slider("Confidence threshold", 0.05, 0.90, 0.25, 0.05)
        image_size = st.select_slider("Inference image size", options=[416, 512, 640, 768, 960], value=640)

        st.header("Density thresholds")
        low_max = st.number_input("Low max count", min_value=1, max_value=100, value=5)
        medium_max = st.number_input("Medium max count", min_value=int(low_max) + 1, max_value=150, value=12)
        st.caption(density_threshold_description(low_max_count=int(low_max), medium_max_count=int(medium_max)))

    uploaded_files = st.file_uploader(
        "Upload traffic images",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Upload one or more traffic-camera images to run vehicle detection.")
        return

    model = cached_model(weights)
    output_root = Path("outputs/streamlit")
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for uploaded_file in uploaded_files:
        suffix = Path(uploaded_file.name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = Path(tmp.name)

        prediction = predict_image(
            tmp_path,
            model=model,
            weights=weights,
            conf=confidence,
            imgsz=int(image_size),
            low_max_count=int(low_max),
            medium_max_count=int(medium_max),
        )
        annotated_path = output_root / f"{Path(uploaded_file.name).stem}_detections.jpg"
        draw_prediction(prediction, annotated_path)
        rows.append(prediction.to_summary_row(str(annotated_path)))

        st.subheader(uploaded_file.name)
        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Detected vehicles", prediction.total_vehicles)
        metric2.metric("Detected-count density", prediction.density_label)
        mean_conf = prediction.mean_vehicle_confidence
        metric3.metric("Mean confidence", "n/a" if mean_conf is None else f"{mean_conf:.2f}")

        if prediction.total_vehicles <= int(low_max) and (mean_conf is None or mean_conf < 0.60):
            st.warning(
                "This density label is based only on detected vehicle boxes. "
                "If the image visibly contains many small or distant vehicles, lower the confidence threshold "
                "to 0.10-0.15, increase image size, or switch to the YOLO11s COCO traffic-general baseline."
            )

        col1, col2 = st.columns(2)
        with col1:
            st.image(Image.open(tmp_path), caption="Original image", use_container_width=True)
        with col2:
            st.image(Image.open(annotated_path), caption="Detected and labeled vehicles", use_container_width=True)

        if prediction.vehicle_counts:
            count_df = pd.DataFrame(
                [{"vehicle class": name, "count": count} for name, count in prediction.vehicle_counts.items()]
            )
            st.dataframe(count_df, hide_index=True, use_container_width=True)
        else:
            st.write("No vehicle-class detections were counted.")
        st.caption(prediction.density_note)

    if rows:
        summary = pd.DataFrame(rows)
        st.subheader("Batch summary")
        st.dataframe(summary, use_container_width=True, hide_index=True)
        csv_bytes = summary.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV summary", csv_bytes, "traffic_vehicle_predictions.csv", "text/csv")


if __name__ == "__main__":
    main()
