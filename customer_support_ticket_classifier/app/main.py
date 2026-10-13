from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# 2. Load DistilBERT models
# --------------------------------------------------

def load_classifier(target):
    model_path = MODEL_DIR / f"distilbert_{target}"

    if not model_path.is_dir():
        raise FileNotFoundError(
            f"DistilBERT model not found: {model_path}"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path
    )

    model.to(DEVICE)
    model.eval()

    return tokenizer, model


# --------------------------------------------------
# 3. FastAPI startup and shutdown
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load both models once at application startup
    app.state.category_tokenizer, app.state.category_model = (
        load_classifier("category")
    )

    app.state.priority_tokenizer, app.state.priority_model = (
        load_classifier("priority")
    )

    app.state.models_loaded = True

    yield

    app.state.models_loaded = False


app = FastAPI(
    title="Customer Support Ticket Classifier",
    description="Classify support tickets by category and priority.",
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------
# 4. Request and response schemas
# --------------------------------------------------

class TicketRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Customer support ticket text",
        examples=["I was charged twice for my subscription"],
    )

    @field_validator("text")
    @classmethod
    def validate_ticket_text(cls, value):
        value = value.strip()

        if not value:
            raise ValueError(
                "Ticket text cannot be empty or whitespace only"
            )

        return value


class TicketResponse(BaseModel):
    text: str
    category: str
    priority: str
    confidence: float
    category_confidence: float
    priority_confidence: float


# --------------------------------------------------
# 5. Prediction helper
# --------------------------------------------------

def predict_one(text, tokenizer, model):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=160,
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0]

    predicted_id = int(probabilities.argmax().item())

    label = model.config.id2label[predicted_id]
    confidence = float(probabilities[predicted_id].item())

    return str(label), confidence


# --------------------------------------------------
# 6. API endpoints
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Customer Support Ticket Classifier API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health(request: Request):
    loaded = getattr(
        request.app.state,
        "models_loaded",
        False,
    )

    return {
        "status": "healthy" if loaded else "starting",
        "models_loaded": loaded,
    }


@app.post("/predict", response_model=TicketResponse)
def predict_ticket(
    ticket: TicketRequest,
    request: Request,
):
    if not getattr(
        request.app.state,
        "models_loaded",
        False,
    ):
        raise HTTPException(
            status_code=503,
            detail="Models are not loaded",
        )

    text = ticket.text

    try:
        category, category_confidence = predict_one(
            text,
            request.app.state.category_tokenizer,
            request.app.state.category_model,
        )

        priority, priority_confidence = predict_one(
            text,
            request.app.state.priority_tokenizer,
            request.app.state.priority_model,
        )

        return TicketResponse(
            text=text,
            category=category,
            priority=priority,
            confidence=round(
                min(category_confidence, priority_confidence),
                4,
            ),
            category_confidence=round(
                category_confidence, 4
            ),
            priority_confidence=round(
                priority_confidence, 4
            ),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the model files.",
        ) from exc