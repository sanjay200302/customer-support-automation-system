from pyspark.sql import SparkSession
import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from reply_agent import ReplyAgent
from escalation import EscalationAgent

# Step 1: Connect to Databricks Delta table
spark = SparkSession.builder.appName("RealtimePipeline").getOrCreate()
df = spark.table("workspace.hiver-sde-intern.support_cleaned")
pandas_df = df.toPandas()

# Step 2: Load DistilBERT model
model_path = r"D:\Hiver Assigement\Hiver-SDE-Intern-Project\notebooks\models\intents"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict_intent(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = logits.argmax().item()
    return model.config.id2label[predicted_class_id]

# Step 3: Initialize agents
reply_agent = ReplyAgent()
escalation_agent = EscalationAgent()

# Step 4: Run pipeline
results = []
for _, row in pandas_df.iterrows():
    text = row["clean_text"]
    pred_intent = predict_intent(text)
    decision, reason = escalation_agent.decide(pred_intent)
    reply = reply_agent.generate(pred_intent, text, decision)

    results.append({
        "text": text,
        "intent_pred": pred_intent,
        "escalation_pred": decision,
        "reply": reply,
        "reason": reason
    })

# Step 5: Save results
results_df = pd.DataFrame(results)
results_df.to_csv("realtime_full_pipeline_results.csv", index=False)

print("✅ Real-time pipeline results saved to realtime_full_pipeline_results.csv")
