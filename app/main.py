from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import mysql.connector
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import json
import spacy


# Load the model
model_ckpt = "./model"

config = AutoConfig.from_pretrained(model_ckpt)
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, config=config)
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True, device=-1)

# DB setup
db = mysql.connector.connect(
    host="db",
    user="mluser",
    password="mlpass",
    database="ml_app"
)
cursor = db.cursor()

# Create table if it doesn't exist
cursor.execute("""
CREATE TABLE IF NOT EXISTS classifications (
  id INT AUTO_INCREMENT PRIMARY KEY,
  article TEXT NOT NULL,
  preprocessed TEXT NOT NULL,
  prediction JSON NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")
db.commit()

# FastAPI app
app = FastAPI(title="Math Articles Classifier")
# Templates
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
# Spacy setup
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

@app.get("/", response_class=HTMLResponse)
# Home route
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Classification route
@app.post("/classify", response_class=HTMLResponse)
def classify(request: Request, text: str = Form(...)):
    # Text preprocessing
    doc = nlp(text)

    tokens = []
    for token in doc:
        # Skip stopwords if requested
        if token.is_stop:
            continue
        # Skip punctuation if requested
        if token.is_punct:
            continue
        if token.like_url or token.like_email:
            continue
        # Keep the lowercase version
        tokens.append(token.text.lower())
    
    preprocessed = " ".join(tokens)

    # Run prediction
    predictions = pipe(preprocessed, truncation=True, padding=True, max_length=512)
    preds = {}

    for i in predictions[0]:
        if i["score"] > 0.5:
            preds[i["label"]] = i["score"]
        else:
            pass
    
    # Prepare output
    if preds:
        output = list(preds.keys())
    else:
        # Return "No confident prediction" if no label was assigned with >50% confidence
        output = ["No confident prediction"]
    

    # Save to DB
    cursor.execute(
        "INSERT INTO classifications (article, preprocessed, prediction) VALUES (%s, %s, %s)",
        (text, preprocessed, json.dumps(preds))
    )
    db.commit()
    
    return templates.TemplateResponse(
        "result.html",
        {"request": request, "article": text, "preprocessed": preprocessed, "output": output}
    )

# History route
@app.get("/history", response_class=HTMLResponse)
def history(request: Request):
    # Fetch last 20 classifications
    cursor.execute("SELECT id, article, preprocessed, prediction, created_at FROM classifications ORDER BY created_at DESC LIMIT 20")
    rows = cursor.fetchall()

    # Format rows for display
    formatted_rows = [
        {
            "id": r[0],
            "article": r[1],
            "preprocessed": r[2],
            "prediction": json.loads(r[3]),
            "created_at": r[4]
        }
        for r in rows
    ]

    return templates.TemplateResponse("history.html", {"request": request, "rows": formatted_rows})