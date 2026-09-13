import pandas as pd

# Load evaluation results
results_full = pd.read_csv("results_full.csv")

# Load reply logs
reply_log = pd.read_csv("reply_log.csv")

# Rename reply_log columns to match results_full
reply_log = reply_log.rename(columns={
    "intent": "intent_pred",
    "customer_text": "text",
    "decision": "escalation_pred"
})

# Merge on predicted intent + text + escalation decision
merged = pd.merge(
    results_full,
    reply_log,
    on=["intent_pred", "text", "escalation_pred"],
    how="left"
)

# Save consolidated file
merged.to_csv("consolidated_results.csv", index=False)

print(" Consolidated results saved to consolidated_results.csv")
