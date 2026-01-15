from flask import Flask, request, jsonify
from flask_cors import CORS

from predict import HateSpeechPredictor, censor_text

app = Flask(__name__)
CORS(app)

# Load model ONCE when server starts
predictor = HateSpeechPredictor()

@app.get("/")
def home():
    return "Hate Speech API is running. Try /health or POST /predict"

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

@app.post("/predict")
def predict():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    remove = bool(data.get("remove", False))

    if not text:
        return jsonify({"error": "Text is required"}), 400

    out = predictor.predict(text)
    cleaned = censor_text(text, out["flagged_words"]) if remove else text

    return jsonify({
        "label": out["label"],
        "confidence": out["confidence"],
        "flagged_words": out["flagged_words"],
        "cleaned_text": cleaned
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8001, debug=True)
