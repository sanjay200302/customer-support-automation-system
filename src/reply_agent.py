from transformers import pipeline
import pandas as pd
import os

class ReplyAgent:
    def __init__(self, model_name="google/flan-t5-small", use_gpu=True, log_file="reply_log.csv"):
        # Templates for common intents (tuned for politeness + brand tone)
        self.templates = {
            "delivery_delay": "We’re sorry your package is delayed. Please DM us your order ID so we can assist you promptly.",
            "order_issue": "I’m sorry you received the wrong item. Kindly share your order details via DM so we can resolve this quickly.",
            "billing_problem": "We understand billing issues are sensitive. Please DM us your account details securely so our support team can assist.",
            "account_access": "Account access problems require special handling. We’ve escalated your issue to our account specialists.",
            "complaint_unresolved": "We regret your issue remains unresolved. We’ve escalated this to our support team for immediate attention.",
            "praise": "Thank you so much for your kind words! We truly appreciate your support.",
            "product_query": "We’d be happy to answer your product query. Could you DM us the product details so we can assist?",
            "return_refund": "We’re truly sorry for the inconvenience. Please DM us your order ID so we can assist you with the return/refund."
        }

        # Use GPU if available
        device = 0 if use_gpu else -1
        self.generator = pipeline("text2text-generation", model=model_name, device=device)

        # Logging setup
        self.log_file = log_file
        if not os.path.exists(self.log_file):
            pd.DataFrame(columns=["intent", "customer_text", "decision", "reply"]).to_csv(self.log_file, index=False)

    def generate(self, intent, customer_text, escalation_decision):
        """
        Generate a reply based on intent and escalation decision.
        Uses templates first, falls back to LLM if intent is unknown.
        Logs every reply into a CSV file.
        """
        if escalation_decision == "auto":
            if intent in self.templates:
                reply = self.templates[intent]
            else:
                prompt = f"Write a polite AmazonHelp reply for intent '{intent}' to: {customer_text}"
                reply = self.generator(
                    prompt,
                    max_length=60,
                    do_sample=False,
                    num_return_sequences=1
                )[0]["generated_text"]
        else:
            reply = "We’ve escalated your issue to our support team. They’ll follow up with you shortly. Thank you for your patience."

        # Log reply
        self._log_reply(intent, customer_text, escalation_decision, reply)

        return reply

    def _log_reply(self, intent, customer_text, decision, reply):
        """Append reply details to CSV log file."""
        df = pd.DataFrame([{
            "intent": intent,
            "customer_text": customer_text,
            "decision": decision,
            "reply": reply
        }])
        df.to_csv(self.log_file, mode="a", header=False, index=False)
