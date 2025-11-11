import os
import pandas as pd
import abc
import requests
import json
import sys

class LLM(metaclass=abc.ABCMeta):
    DEFAULT_SYSTEM_PROMPT = (
        "Each dialog reflects an interaction between a travel agent and a customer discussing flights, hotels, "
        "or other aspects of a trip. Your task is to be in the role of the customer, for example, to find the best option for your trip. "
        "Using the dialog history, finish the dialog with one line as you see fit on behalf of the customer (user). Derive the result. \n"
    )

    def __init__(self, dialog_file):
        self.dialog_df = pd.read_csv(dialog_file)

    @abc.abstractmethod
    def generate(self, messages, dialog_id, context):
        pass

    def prepare_dialogues(self):
        dialogues = []
        for dialog_id, group in self.dialog_df.groupby("dialog_id"):
            group = group.sort_values(by="turn").head(10)  # Limit to first n lines
            context = "\n".join(group["context"].tolist())
            messages = [{"role": "system", "content": self.DEFAULT_SYSTEM_PROMPT}]
            messages.append({"role": "user", "content": context})
            dialogues.append((dialog_id, context, messages))
        return dialogues

class OllamaLLM(LLM):
    def __init__(self, dialog_file):
        super().__init__(dialog_file)
        self.model_name = "dolphin-llama3"

    def generate(self, messages, dialog_id, context):
        payload = {
            "model": self.model_name,
            "stream": False,
            "messages": messages,
            "options": {"top_k": 1, "num_predict": 128, "num_ctx": 2048, "temperature": 1.0}
        }
        response = requests.post("http://localhost:11434/api/chat", json=payload)
        if response.status_code == 200:
            response_content = response.json()
            content = response_content.get("message", {}).get("content", "")
            return {"dialog_id": dialog_id, "context": context, "generated_response": content}
        else:
            raise ConnectionError(f"Error: {response.text}")

# File Paths
dialog_file = "C:/Users/daren/Desktop/MSc Thesis Bau/thesis-zhukova/FramesExp2/ultimate_test.csv"
#dialog_file = "/home/zeso5934/thesis-zhukova/FramesExp2/ultimate_test.csv"

# Initialize LLM
llm = OllamaLLM(dialog_file)

dialogues = llm.prepare_dialogues()
results = []

# Generate responses
for dialog_id, context, messages in dialogues:
    response = llm.generate(messages, dialog_id, context)
    results.append(response)

# Save Results
#output_dir = "/home/zeso5934/thesis-zhukova/FramesExp2"
output_dir = "C:/Users/daren/Desktop/MSc Thesis Bau/thesis-zhukova/FramesExp2"

os.makedirs(output_dir, exist_ok=True)

results_df = pd.DataFrame(results)
output_file = os.path.join(output_dir, "2703_dolphin-llama3_10.csv")
results_df.to_csv(output_file, index=False)

print("Generated responses saved to", output_file)
