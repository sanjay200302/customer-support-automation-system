import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("results_full.csv")


# Confusion Matrix
plt.figure(figsize=(6,6))
sns.heatmap(pd.crosstab(df["intent_pred"], df["escalation_pred"]), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# Escalation Distribution
plt.figure(figsize=(6,4))
sns.countplot(x="escalation_pred", data=df)
plt.title("Escalation Distribution")
plt.show()

# Reply Length Boxplot (proxy for quality)
df["reply_length"] = df["reply"].apply(len)
plt.figure(figsize=(6,4))
sns.boxplot(x="escalation_pred", y="reply_length", data=df)
plt.title("Reply Quality (Length Proxy)")
plt.show()
