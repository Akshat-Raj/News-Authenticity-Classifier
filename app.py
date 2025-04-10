from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os

app = Flask(__name__)

model_path = "saved_model"
pt_file = os.path.join(model_path, "saved_model.pt")

# Use appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load model architecture
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)

# Load fine-tuned weights
model.load_state_dict(torch.load(pt_file, map_location=device))
model.to(device)
model.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text'].strip()

    # Length check
    if len(text) == 0:
        return jsonify({
            "error": "Empty input text.",
            "prediction": None,
            "confidence": None
        }), 400

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model(**inputs)
        logits = output.logits
        probability = torch.sigmoid(logits).item()
        label = "FAKE" if probability >= 0.5 else "REAL"
    
    return jsonify({
        "prediction": label,
        "confidence": f"{probability:.4f}"
    })

if __name__ == '__main__':
    app.run(debug=True)
