
from pathlib import Path
import urllib.request

import cv2
import numpy as np
from flask import Flask, render_template, request, send_from_directory, flash, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "image-colorization-local-app"

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
INPUT_DIR = BASE_DIR / "data" / "input"
OUTPUT_DIR = BASE_DIR / "data" / "output"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}

MODEL_URLS = {
    "colorization_deploy_v2.prototxt":
        "https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/models/colorization_deploy_v2.prototxt",
    "colorization_release_v2.caffemodel":
        "https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/models/colorization_release_v2.caffemodel",
    "pts_in_hull.npy":
        "https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/resources/pts_in_hull.npy",
}


def download_models():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for filename, url in MODEL_URLS.items():
        path = MODEL_DIR / filename

        if not path.exists() or path.stat().st_size == 0:
            print(f"Downloading model: {filename}")
            urllib.request.urlretrieve(url, path)

    return True


def colorize_image(input_path, output_path):
    download_models()

    net = cv2.dnn.readNetFromCaffe(
        str(MODEL_DIR / "colorization_deploy_v2.prototxt"),
        str(MODEL_DIR / "colorization_release_v2.caffemodel"),
    )

    points = np.load(MODEL_DIR / "pts_in_hull.npy")
    points = points.transpose().reshape(2, 313, 1, 1).astype(np.float32)

    net.getLayer(net.getLayerId("class8_ab")).blobs = [points]
    net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [
        np.full((1, 313), 2.606, dtype=np.float32)
    ]

    image = cv2.imread(str(input_path))

    if image is None:
        raise ValueError("Invalid image. Please upload a valid photo.")

    image = image.astype(np.float32) / 255.0
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    original_l = lab_image[:, :, 0].copy()

    resized = cv2.resize(lab_image, (224, 224))
    l_channel = resized[:, :, 0]
    l_channel -= 50

    blob = cv2.dnn.blobFromImage(l_channel)
    net.setInput(blob)

    predicted_ab = net.forward()[0].transpose((1, 2, 0))

    predicted_ab = cv2.resize(
        predicted_ab,
        (image.shape[1], image.shape[0])
    )

    colorized_lab = np.concatenate(
        (original_l[:, :, np.newaxis], predicted_ab),
        axis=2
    )

    result = cv2.cvtColor(colorized_lab, cv2.COLOR_LAB2BGR)
    result = np.clip(result * 255, 0, 255).astype(np.uint8)

    if not cv2.imwrite(str(output_path), result):
        raise RuntimeError("Unable to save the colorized image.")


@app.route("/", methods=["GET", "POST"])
def index():
    original = None
    result = None

    if request.method == "POST":
        uploaded_file = request.files.get("image")

        if not uploaded_file or not uploaded_file.filename:
            flash("Please select an image.")
            return redirect(url_for("index"))

        filename = secure_filename(uploaded_file.filename)

        if "." not in filename or filename.rsplit(".", 1)[1].lower() not in ALLOWED_EXTENSIONS:
            flash("Upload PNG, JPG, JPEG, BMP, or WEBP images only.")
            return redirect(url_for("index"))

        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        input_path = INPUT_DIR / filename
        output_name = f"colorized_{Path(filename).stem}.png"
        output_path = OUTPUT_DIR / output_name

        uploaded_file.save(input_path)

        try:
            colorize_image(input_path, output_path)

            original = url_for(
                "uploaded_file",
                folder="input",
                filename=filename
            )
            result = url_for(
                "uploaded_file",
                folder="output",
                filename=output_name
            )

        except Exception as error:
            app.logger.exception("Colorization failed")
            flash(f"Colorization failed: {error}")

    return render_template(
        "index.html",
        original=original,
        result=result
    )


@app.route("/files/<folder>/<path:filename>")
def uploaded_file(folder, filename):
    if folder == "input":
        directory = INPUT_DIR
    elif folder == "output":
        directory = OUTPUT_DIR
    else:
        return "Invalid folder", 404

    return send_from_directory(directory, filename)


if __name__ == "__main__":
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    app.run(debug=True)
