import pandas as pd
import abc
import requests
import json

class LLM(metaclass=abc.ABCMeta):
    DEFAULT_SYSTEM_PROMPT = """There is a dialog between you and the user, these are your last responses. 
    Try to continue the dialog in the same way. Start the turn with a decision."""

    def __init__(self, full_dialog_file, positive_outcomes_file, negative_outcomes_file):
        self.full_dialog_df = pd.read_csv(full_dialog_file)
        self.positive_B2_numbers = pd.read_csv(positive_outcomes_file)['B2'].tolist()
        self.negative_filtered_df = pd.read_csv(negative_outcomes_file)
        self.negative_B2_numbers = self.negative_filtered_df[
            (self.negative_filtered_df['Donated'] == 'no') & 
            (self.negative_filtered_df['Context'] != 'stop')
        ]['B2'].tolist()

    @abc.abstractmethod
    def generate(self, messages, B2, context):
        pass

    def prepare_dialogues(self, B2_numbers):
        dialogues = []
        for B2_number in B2_numbers:
            dialog = self.full_dialog_df[self.full_dialog_df['B2'] == B2_number]
            dialog = dialog.head(15)  # Take only the first n lines
            user_lines = dialog[dialog['B4'] == 0]['Unit'].tolist()
            assistant_lines = dialog[dialog['B4'] == 1]['Unit'].tolist()
            
            messages = []
            for user_line, assistant_line in zip(user_lines, assistant_lines):
                messages.append({"role": "user", "content": user_line})
                messages.append({"role": "assistant", "content": assistant_line})

            context = ' '.join(dialog['Unit'].tolist())  # Create context string from dialog lines
            dialogues.append((B2_number, context, messages))

        return dialogues


class OllamaLLM(LLM):
    def generate(self, messages, B2, context):
        payload = {
            "model": "mistral",
            "prompt": self.DEFAULT_SYSTEM_PROMPT,
            "stream": False,
            "top_k": 1,  # Limit generation to top 1 result
            "messages": messages
        }

        response = requests.post("http://localhost:11434/api/chat", json=payload)
        #response = requests.post("https://llm.srv.webis.de/api/chat", json=payload)

        if response.status_code == 200:
            response_content = response.json()
            content = response_content.get('choices', [{}])[0].get('message', {}).get('content', '')
            print(response.text)  
            print(content) # Print the response content
            classification = self.classify_response(content)
            return {"B2": B2, "context": context, "decision": classification}
        else:
            raise ConnectionError(f"Error: {response.text}")

    def classify_response(self, content):
        negative_tokens = ["zero", "never", "absolutely not", "0", "no", "can't", "won't"]
        for token in negative_tokens:
            if token in content.lower():
                return "negative"
        return "positive/undecided"


full_dialog_file = '/Users/daren/Desktop/MSc Thesis Bau/thesis-zhukova/full_dialog.csv'
positive_outcomes_file = '/Users/daren/Desktop/MSc Thesis Bau/thesis-zhukova/rnd_B2_B6_gr0.csv'
negative_outcomes_file = '/Users/daren/Desktop/MSc Thesis Bau/thesis-zhukova/B60 neg+pos outcomes.csv'

llm = OllamaLLM(full_dialog_file, positive_outcomes_file, negative_outcomes_file)

positive_dialogues = llm.prepare_dialogues(llm.positive_B2_numbers)
negative_dialogues = llm.prepare_dialogues(llm.negative_B2_numbers)

results = []

for B2, context, messages in positive_dialogues:
    response = llm.generate(messages, B2, context)
    results.append(response)

for B2, context, messages in negative_dialogues:
    response = llm.generate(messages, B2, context)
    results.append(response)

# Save results to a CSV file in the specified folder
results_df = pd.DataFrame(results)
results_df.to_csv('/Users/daren/Desktop/MSc Thesis Bau/thesis-zhukova/result_table.csv', index=False)

# Save all responses to a JSON file
with open('response_10lines.json', 'w') as f:
    json.dump(results, f)

print("Results saved to result_table.csv")
print("Responses saved to response_15lines.json")
