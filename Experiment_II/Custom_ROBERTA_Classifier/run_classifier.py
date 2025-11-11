import pickle

import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification


def main():
    # load binarizer
    with open("models/mlb.pkl", "rb") as f:
        mlb = pickle.load(f)

    # load models
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('models/checkpoint-7440')
    model.eval()

    example = "Thank you! Goodbye!"

    # example inference
    inputs = tokenizer(example, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    label_ids = torch.sigmoid(logits).round().detach()
    labels = mlb.inverse_transform(label_ids)
    print(labels)
    # [('goodbye', 'thankyou')]

if __name__ == '__main__':
    main()
