class EscalationAgent:
    def __init__(self):
        # Define which intents can be auto-handled
        self.auto_intents = {
            "delivery_delay": "Routine delivery delays can be auto-handled with tracking info.",
            "praise": "Positive feedback can be acknowledged automatically.",
            "product_query": "Simple product queries can be answered automatically.",
            "return_refund": "Return/refund requests can be initiated automatically."
        }

        # Define which intents require escalation
        self.escalate_intents = {
            "billing_problem": "Billing issues require secure handling by human support.",
            "account_access": "Account access problems involve sensitive data and must be escalated.",
            "complaint_unresolved": "Unresolved complaints need human intervention to ensure resolution.",
            "order_issue": "Wrong or missing items often need manual support to resolve."
        }

    def decide(self, intent):
        if intent in self.auto_intents:
            return "auto", self.auto_intents[intent]
        elif intent in self.escalate_intents:
            return "escalate", self.escalate_intents[intent]
        else:
            return "escalate", "Unknown intent — escalated by default for safety."
