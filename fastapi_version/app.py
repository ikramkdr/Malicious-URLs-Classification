from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import torch
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
import uuid
from models.mlp_model import MLPClassifier
from utils.generate_features import extract_features_from_url
from fastapi.staticfiles import StaticFiles
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

def load_urls():
    with open("test_urls.txt", "r", encoding="utf-8", errors="ignore") as f:
        return [line.strip() for line in f.readlines()]

urls = load_urls()

# Charging the scaler
scaler = joblib.load("scaler.pkl")
input_dim = len(extract_features_from_url(urls[0]))

# Charging the model
mlp_model = MLPClassifier(input_dim)
mlp_model.load_state_dict(torch.load("mlp_model.pt"))
mlp_model.eval()

rf_model = joblib.load("rf_model.pkl")

@app.get("/", response_class=HTMLResponse)
def landing(request: Request):
    return templates.TemplateResponse("landing.html", {"request": request})

@app.get("/predictor", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "urls": urls,
        "selected": None,
        "result": None,
        "img": None,
        "model_choice": None
    })

@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    selected_url: str = Form(""),
    model_choice: str = Form("mlp")
):
    # Choosing the URL 
    url_to_analyze = selected_url


    # White list of known benign domains
    whitelist = [
        "youtube.com", "www.youtube.com", "google.com", "www.google.com",
        "facebook.com", "www.facebook.com", "github.com", "www.github.com"
    ]

    # Si l'URL est whitelistée, on force le résultat
    if url_to_analyze.lower() in whitelist:
        label = (
            f"Model used: {'MLP ' if model_choice == 'mlp' else 'Random Forest '}<br>"
            f"<strong>This url has a known benign domain:</strong> {url_to_analyze}<br>"
            f"Final decision: Benign ✅ (whitelist)"
        )
        img_filename = None
    else:
        # if not: Normal prediction
        features = extract_features_from_url(url_to_analyze)
        df_feat = pd.DataFrame([features])
        X_scaled = scaler.transform(df_feat)

        if model_choice == "mlp":
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            with torch.no_grad():
                output = mlp_model(X_tensor)
                probs = torch.softmax(output, dim=1).numpy()[0]
        else:
            probs = rf_model.predict_proba(X_scaled)[0]

        benign_score = round(float(probs[0]), 4)
        malicious_score = round(float(probs[1]), 4)
        pred = int(probs.argmax())

        label = (
            f"Model used: {'MLP ' if model_choice == 'mlp' else 'Random Forest '}<br>"
            f"Benign : {benign_score}<br>"
            f"Malicious : {malicious_score}<br>"
            f"<strong>Final decision:</strong> {'BENIGN' if pred == 0 else 'MALICIOUS!!'}"
        )

        # Delete old images before generating a new one
        for filename in os.listdir("static"):
            if filename.endswith(".png"):
                try:
                    os.remove(os.path.join("static", filename))
                except:
                    pass

        # Creating the graph
        img_id = str(uuid.uuid4())
        img_path = f"static/{img_id}.png"

        plt.figure(figsize=(4, 3))
        plt.bar(["Benign ", "Malicious "], [benign_score, malicious_score], color=["blue", "red"])
        plt.ylim(0, 1)
        plt.title("Prediction Scores")
        plt.tight_layout()
        plt.savefig(img_path)
        plt.close()

        img_filename = f"{img_id}.png"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "urls": urls,
        "selected": url_to_analyze,
        "result": label,
        "img": img_filename,
        "model_choice": model_choice
    })