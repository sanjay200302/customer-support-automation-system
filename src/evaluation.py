import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

class EvaluationHarness:
    def __init__(self, labels):
        self.labels = labels

    # -------------------------
    # Intent evaluation
    # -------------------------
    def evaluate_intents(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted")
        cm = confusion_matrix(y_true, y_pred, labels=self.labels)
        return {"accuracy": acc, "f1": f1, "confusion_matrix": cm}

    # -------------------------
    # Escalation evaluation
    # -------------------------
    def evaluate_escalation(self, y_true, y_pred):
        precision = precision_score(y_true, y_pred, pos_label="escalate")
        recall = recall_score(y_true, y_pred, pos_label="escalate")
        return {"precision": precision, "recall": recall}

    # -------------------------
    # Reply quality rubric
    # -------------------------
    def judge_reply(self, reply_text):
        """
        Rubric scoring for reply quality.
        Returns politeness, conciseness, and brand tone scores (0–1).
        """

        # Politeness: check for polite words
        polite_words = ["sorry", "thank", "appreciate", "please"]
        politeness = sum(word in reply_text.lower() for word in polite_words) / len(polite_words)

        # Conciseness: penalize if reply is too long (>40 words)
        word_count = len(reply_text.split())
        conciseness = 1.0 if word_count <= 40 else max(0.0, 1 - (word_count - 40) / 40)

        # Brand tone: check for AmazonHelp style words
        brand_words = ["dm", "assist", "support", "help"]
        brand_tone = sum(word in reply_text.lower() for word in brand_words) / len(brand_words)

        return {
            "politeness": round(politeness, 2),
            "conciseness": round(conciseness, 2),
            "brand_tone": round(brand_tone, 2)
        }

    # -------------------------
    # Dataset integration
    # -------------------------
    def load_csv(self, path):
        """
        Load golden set from CSV.
        Expects columns: clean_text, intent, escalation_decision
        """
        df = pd.read_csv(path)
        return df
