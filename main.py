from flask import Flask, jsonify, request
from service import (
    model_init_opanai,
    model_init_api,
    load_config,
    model_predict,
    graph_init,
    model_predict_image,
)
import base64
from pathlib import Path

app = Flask(__name__)


@app.route("/")
def say_polo():
    return jsonify({"message": "polo"})


@app.route("/predict", methods=["GET"])
def predict():
    input_text = request.args.get("text", "")
    if not input_text:
        return jsonify({"error": "Missing required parameter: text"}), 400
    result = model_predict(input_text)
    return jsonify({"result": result})


@app.route("/predict_image", methods=["POST"])
def predict_image():
    input_text = request.form.get("text", "")
    if not input_text:
        return jsonify({"error": "Missing required parameter: text"}), 400

    image_b64 = None
    media_type = None
    file = request.files.get("image")

    if file and file.filename:
        suffix = Path(file.filename).suffix.lower()
        media_type_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        media_type = media_type_map.get(suffix, "image/jpeg")
        image_b64 = base64.standard_b64encode(file.read()).decode("utf-8")

    result = model_predict_image(input_text, image_b64=image_b64, media_type=media_type)
    return jsonify({"result": result})


if __name__ == "__main__":
    config = load_config()
    if config["server"]["model"] == "openai":
        model_init_opanai()
        graph_init()
    elif config["server"]["model"] == "llama_api":
        model_init_api()
        graph_init()
    app.run(
        host=config["server"]["host"],
        port=config["server"]["port"],
        debug=config["server"]["debug"],
    )
