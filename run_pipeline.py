import sys, os
import torch
import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# --- Ensure src folder is always found ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(BASE_DIR, "src")
sys.path.append(SRC_PATH)

from reply_agent import ReplyAgent
from escalation import EscalationAgent
from evaluation import EvaluationHarness

# --- Step 1: Initialize agents ---
reply_agent = ReplyAgent()
escalation_agent = EscalationAgent()
evaluator = EvaluationHarness(labels=[
    "delivery_delay","order_issue","billing_problem","account_access",
    "complaint_unresolved","praise","product_query","return_refund"
])

# --- Step 2: Load golden set from CSV ---
csv_path = r"D:\Hiver Assigement\Hiver-SDE-Intern-Project\notebooks\data\golden_set_balanced.csv"
df = evaluator.load_csv(csv_path)

# For testing on subset, uncomment:
# df = df.sample(50, random_state=42)

# --- Step 3: Load DistilBERT model ---
model_path = r"D:\Hiver Assigement\Hiver-SDE-Intern-Project\notebooks\models\intents"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict_intent(text):
    """Predict intent using DistilBERT classifier"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = logits.argmax().item()
    return model.config.id2label[predicted_class_id]

# --- Step 4: Run pipeline on dataset ---
y_true, y_pred, esc_true, esc_pred = [], [], [], []
results = []

for _, row in df.iterrows():
    text = row["clean_text"]
    intent = row["intent"]
    escalation_decision = row["escalation_decision"]

    # Ground truth
    y_true.append(intent)
    esc_true.append(escalation_decision)

    # Predicted intent from DistilBERT
    pred_intent = predict_intent(text)
    y_pred.append(pred_intent)

    # Escalation agent decision (based on predicted intent)
    decision, reason = escalation_agent.decide(pred_intent)
    esc_pred.append(decision)

    # Generate reply
    reply = reply_agent.generate(pred_intent, text, decision)

    # Evaluate reply quality
    scores = evaluator.judge_reply(reply)

    # Save results
    results.append({
        "text": text,
        "intent_true": intent,
        "intent_pred": pred_intent,
        "escalation_true": escalation_decision,
        "escalation_pred": decision,
        "reply": reply,
        "scores": scores
    })

    # Print to console
    print(f"Intent: {intent} | Predicted: {pred_intent}\nCustomer: {text}\nDecision: {decision}\nReason: {reason}\nReply: {reply}\nScores: {scores}\n{'-'*60}")

# --- Step 5: Evaluate overall performance ---
print("Intent Evaluation:", evaluator.evaluate_intents(y_true, y_pred))
print("Escalation Evaluation:", evaluator.evaluate_escalation(esc_true, esc_pred))

# --- Step 6: Save results to CSV ---
df_results = pd.DataFrame(results)
df_results.to_csv("results_full.csv", index=False)

# Save misclassifications separately
errors = df_results[df_results["intent_true"] != df_results["intent_pred"]]
errors.to_csv("errors.csv", index=False)

print(" Results saved to results_full.csv and errors.csv")
