# Decision Log

Below are the 15 non‑obvious decisions made during the project, with reasoning.

1. **Chose DistilBERT for intent classification**  
   DistilBERT was selected because it provides strong contextual understanding while being lightweight enough to train on a mid‑range GPU. This ensured reproducibility within the assignment’s 15‑minute runtime constraint.

2. **Balanced dataset via upsampling**  
   The golden set was skewed toward majority classes (e.g., order issues). We upsampled minority classes to balance representation, improving weighted F1 and preventing bias toward dominant intents.

3. **Defined 8 core intents**  
   From noisy Twitter data, we distilled intents into 8 categories (account access, billing problem, complaint unresolved, delivery delay, order issue, praise, product query, return/refund). This kept classification tractable while covering the brand’s main support needs.

4. **Hand‑labeled golden set (150–250 examples)**  
   We manually corrected and verified labels to create a reliable evaluation set. This ensured metrics reflected true performance rather than noisy auto‑labels.

5. **Weighted F1 as primary metric**  
   Accuracy alone would inflate results on imbalanced data. Weighted F1 was chosen because it balances precision and recall across all classes, reflecting real support performance.

6. **Evaluation harness with Hugging Face Trainer**  
   Hugging Face’s Trainer standardized training, evaluation, and logging. This reduced boilerplate code and ensured reproducibility with clear metrics.

7. **Baseline comparisons included**  
   We added two baselines: trivial majority class and TF‑IDF + Logistic Regression. This contextualized DistilBERT’s performance and proved it wasn’t just memorizing.

8. **Confusion matrix for failure analysis**  
   We plotted confusion matrices to visualize misclassifications. This revealed that “complaint_unresolved” was sometimes confused with “delivery_delay,” guiding hypotheses about overlapping language.

9. **LLM‑based reply generation pipeline**  
   For reply drafting, we used an LLM pipeline that mimics brand tone. This ensured responses were grounded in historical resolution patterns while remaining flexible.

10. **Escalation rules defined in YAML config**  
   Escalation logic (auto vs. human handoff) was externalized into `brand_config.yaml`. This made rules transparent, editable, and brand‑specific.

11. **Spark preprocessing for raw Kaggle data**  
   PySpark was used to clean and filter brand‑specific tweets (AmazonHelp). This scaled better than pandas for multi‑million tweet datasets and ensured reproducibility.

12. **Reproducibility via configs**  
   Hyperparameters, model paths, and brand rules were stored in YAML configs. This allowed quick reruns and made the pipeline reproducible for reviewers.

13. **Unit tests for robustness**  
   Tests (`test_intents.py`, `test_reply_agent.py`, etc.) validated each module. This ensured the pipeline didn’t silently break and gave confidence in modular reliability.

14. **Saved checkpoints and best model**  
   The Trainer was configured to save checkpoints per epoch and load the best model by F1. This avoided overfitting and ensured we always had the strongest version available.

15. **Decision to subsample dataset for runtime**  
   Instead of training on the full Kaggle dataset (~3M tweets), we subsampled to brand‑specific threads. This kept runtime under 15 minutes, meeting assignment constraints while still demonstrating proof of concept.
