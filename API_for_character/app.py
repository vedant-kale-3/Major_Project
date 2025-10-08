# app.py
import os
from joblib import load
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
VECT_PATH = os.path.join(MODEL_DIR, "vectorizer.joblib")
CLASSES_PATH = os.path.join(MODEL_DIR, "classes.joblib")

app = FastAPI(title="Task Category Classifier")

# Allow broad CORS for development (change in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model artifacts at startup
if not (os.path.exists(MODEL_PATH) and os.path.exists(VECT_PATH) and os.path.exists(CLASSES_PATH)):
    raise FileNotFoundError("Model artifacts not found. Run `python train.py` first to create models/")

model = load(MODEL_PATH)
vectorizer = load(VECT_PATH)
classes = load(CLASSES_PATH)

def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=-1, keepdims=True)

class TaskRequest(BaseModel):
    task: str

class BatchRequest(BaseModel):
    tasks: list[str]

@app.get("/")
def root():
    return {"status": "ok", "detail": "Task Category Classifier API"}

@app.post("/predict")
def predict(req: TaskRequest):
    text = req.task
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Empty task text provided.")

    x_tfidf = vectorizer.transform([text])
    scores = model.decision_function(x_tfidf)  # shape depends on binary vs multiclass

    # Convert scores to a probability-like distribution
    if scores.ndim == 1:
        # Binary case: decision_function returns 1-d array -> apply sigmoid
        sigmoid = lambda s: 1 / (1 + np.exp(-s))
        prob_pos = sigmoid(scores[0])
        # classes order: model.classes_[0], model.classes_[1]
        probs = np.array([1 - prob_pos, prob_pos])
    else:
        # Multiclass: use softmax over the decision scores
        probs = _softmax(scores[0])

    best_idx = int(np.argmax(probs))
    predicted = model.classes_[best_idx]
    confidence = float(probs[best_idx])

    return {
        "task": text,
        "predicted_category": str(predicted),
        "confidence": confidence,
        "all_classes": [str(c) for c in model.classes_],
        "class_confidences": [float(p) for p in probs.tolist()]
    }

@app.post("/predict-batch")
def predict_batch(req: BatchRequest):
    texts = req.tasks
    if not isinstance(texts, list) or len(texts) == 0:
        raise HTTPException(status_code=400, detail="Please provide a non-empty list of tasks.")
    X_tfidf = vectorizer.transform([str(t) for t in texts])
    scores = model.decision_function(X_tfidf)

    results = []
    for i in range(len(texts)):
        s = scores[i]
        if s.ndim if isinstance(s, np.ndarray) else False:
            pass
        if isinstance(s, np.ndarray) and s.ndim == 1:
            probs = _softmax(s)
        else:
            # binary: shape may be scalar; sklearn returns 1-d array for binary
            if scores.ndim == 1:
                sigmoid = lambda s: 1 / (1 + np.exp(-s))
                prob_pos = sigmoid(scores[i])
                probs = np.array([1 - prob_pos, prob_pos])
            else:
                probs = _softmax(np.atleast_1d(s))

        best_idx = int(np.argmax(probs))
        results.append({
            "task": texts[i],
            "predicted_category": str(model.classes_[best_idx]),
            "confidence": float(probs[best_idx]),
            "class_confidences": [float(p) for p in probs.tolist()]
        })

    return {"results": results}
