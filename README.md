# Smart Financial Assistant

20-hour hackathon project — NLP intent classification + rule-based risk detection + FastAPI backend + iOS app.

---

## Quick Start

### 1. Python environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r backend/requirements.txt
```

### 2. Train the ML model
```bash
cd ml
python train.py
```

### 3. Run the backend
```bash
cd backend
uvicorn main:app --reload --port 8000
```
Open http://localhost:8000/docs to test the API.

### 4. Export to Core ML (Mac only)
```bash
cd ml
python export_coreml.py
```
Drag `intent_model.mlmodel` into the Xcode project.

---

## Project Structure

```
smart-financial-assistant/
├── ml/           # Intent classifier (sklearn → Core ML)
├── backend/      # FastAPI server (intent + risk + response)
├── ios/          # Swift iOS app
└── data/         # Sample CSVs for training & testing
```

## API

**POST /chat**
```json
{
  "user_id": "user_001",
  "message": "I see a charge I don't recognize",
  "transaction_amount": 8500,
  "failed_attempts": 3,
  "is_anomaly": true,
  "transaction_country": "US"
}
```

## Team
- Person 1 → ML
- Person 2 → Data & Risk
- Person 3 → Backend
- Person 4 → iOS
