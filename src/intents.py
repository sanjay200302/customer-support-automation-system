from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

class IntentClassifier:
    def __init__(self, model_path="./models/intents"):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model(**inputs)
        pred_id = torch.argmax(outputs.logits, dim=1).item()
        return self.model.config.id2label[pred_id]
