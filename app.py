from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import base64, io, cv2
from PIL import Image
import numpy as np
from collections import Counter

app = Flask(__name__)

# Load YOLO model
model = YOLO("yolov8m.pt")  # use yolov8n/s/m/l depending on size

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]

    # Ensure image is RGB
    image = Image.open(file.stream).convert("RGB")

    # Run YOLO
    results = model.predict(image)

    # Collect labels
    all_labels = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            all_labels.append(label)

    # Count detected objects
    counts = Counter(all_labels)
    text = ", ".join([f"{v} {k}{'s' if v > 1 else ''}" for k, v in counts.items()]) or "No objects detected"

    # Annotated image from YOLO
    annotated = results[0].plot()  # BGR (OpenCV format)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)  # ✅ convert to RGB

    # Convert to PIL for saving
    im_pil = Image.fromarray(annotated_rgb)
    buffer = io.BytesIO()
    im_pil.save(buffer, format="JPEG", quality=95)  # ✅ keep high quality
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return jsonify({"image": img_str, "text": text})

if __name__ == "__main__":
    app.run(debug=True)
