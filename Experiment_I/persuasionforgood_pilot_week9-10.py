import os
import pandas as pd
import abc
import requests
import json
import sys

class LLM(metaclass=abc.ABCMeta):
    DEFAULT_SYSTEM_PROMPT = (
        "Each dialog reflects an interaction between a travel agent and a customer discussing flights, hotels, "
        "or other aspects of a trip. Your task is to be in the role of the customer, for example, to book a ticket, "
        "find a hotel, change a reservation and, in general, find the best option for your trip. Using the dialog history, "
        "finish the dialog as you see fit on behalf of the customer (user). Does the dialog lead to a reservation, "
        "a request for information, or an unresolved issue? Derive the result: \n"
        "1. The client's response; \n"
        "2. The type of Dialog Act (one or more from the list: inform; offer; request; switch frame; suggest; "
        "no result; thankyou; sorry; greeting; affirm; negate; confirm; moreinfo; goodbye; request alts; "
        "request compare; hearmore; you are welcome; canthelp; reject) as you categorize your response."
    )

    def __init__(self, dialog_file):
        self.dialog_df = pd.read_csv(dialog_file)
    
    @abc.abstractmethod
    def generate(self, messages, dialog_id, context):
        pass
    
    def prepare_dialogues(self):
        dialogues = []
        for dialog_id, group in self.dialog_df.groupby("dialog_id"):
            context = "\n".join(group["context"].tolist())
            messages = [{"role": "system", "content": self.DEFAULT_SYSTEM_PROMPT}]
            messages.append({"role": "user", "content": context})
            dialogues.append((dialog_id, context, messages))
        return dialogues

class OllamaLLM(LLM):
    def __init__(self, dialog_file):
        super().__init__(dialog_file)
        self.model_name = "mistral"

    def generate(self, messages, dialog_id, context):
        payload = {
            "model": self.model_name,
            "stream": False,
            "messages": messages,
            "options": {"top_k": 1, "num_predict": 128, "num_ctx": 2048, "temperature": 1.0}
        }
        response = requests.post(
            "http://localhost:11434/api/chat",
            json=payload
        )
        if response.status_code == 200:
            response_content = response.json()
            content = response_content.get("message", {}).get("content", "")
            return {"dialog_id": dialog_id, "context": context, "generated_response": content}
        else:
            raise ConnectionError(f"Error: {response.text}")

# File Paths
dialog_file = "/mnt/data/prep_optimal_test.csv"

# Initialize LLM
llm = OllamaLLM(dialog_file)

dialogues = llm.prepare_dialogues()
results = []

# Generate responses
for dialog_id, context, messages in dialogues:
    response = llm.generate(messages, dialog_id, context)
    results.append(response)

# Save Results
output_dir = "/mnt/data/"
os.makedirs(output_dir, exist_ok=True)

results_df = pd.DataFrame(results)
output_file = os.path.join(output_dir, "generated_dialog_responses.csv")
results_df.to_csv(output_file, index=False)

print("Generated responses saved to", output_file)
