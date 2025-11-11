import datetime
import json
import os.path
import pickle
from copy import deepcopy

import matplotlib.pyplot as plt
import pandas as pd
import torch
import transformers
from datasets import Dataset
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import IterativeStratification
from torch.nn import BCEWithLogitsLoss
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig, TrainingArguments, Trainer, \
    TrainerState, TrainerControl


class LogCallback(transformers.TrainerCallback):
    def __init__(self, trainer):
        super().__init__()
        self._trainer = trainer
        self.init_time = datetime.datetime.now().isoformat()
        self.file_name = f"log-train-{self.init_time}.jsonl"

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        super().on_log(args, state, control, **kwargs)
        with open(os.path.join(args.logging_dir, self.file_name), "a+") as out_file:
            out_file.write(json.dumps(state.log_history[-1]))
            out_file.write("\n")

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy


def train_and_evaluate(model, tokenizer, mlb, train_dataset, val_dataset, num_epochs):
    training_args = TrainingArguments(
        output_dir='./models',
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        learning_rate=5e-5,
        do_train=True,
        do_eval=True,
        eval_strategy='epoch',
        eval_steps=1,
        num_train_epochs=num_epochs,
        logging_strategy='steps',
        logging_dir="./logs",
        logging_steps=10,
        remove_unused_columns=False,
        save_strategy='epoch',
    )

    loss_func = BCEWithLogitsLoss()
    trainer = Trainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_loss_func=lambda outputs, labels, num_items_in_batch: loss_func(outputs.logits, labels),
        compute_metrics=lambda pred: {"f1": f1_score(pred.label_ids, torch.sigmoid(torch.tensor(pred.predictions)).round(), average="macro"),},
        data_collator=lambda d: collate_data(d, tokenizer, mlb)
    )

    trainer.add_callback(LogCallback(trainer))

    trainer.train()


def collate_data(data, tokenizer, mlb):
    texts = [d["context"] for d in data]
    labels = mlb.transform([d["DA_list"] for d in data])

    batch = tokenizer(texts,
            add_special_tokens=True,
            padding="longest",
            truncation=False,
            return_attention_mask=True,
            return_tensors='pt')

    batch["labels"] = torch.tensor(labels, dtype=torch.float32)
    return batch

def k_fold_cross_validation(dataframe, tokenizer, k=5, epochs=10):
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(dataframe['DA_list'])

    stratifier = IterativeStratification(n_splits=k, order=1)
    splits = stratifier.split(dataframe, labels)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for fold_number, (train_index, test_index) in enumerate(splits):
        train_df = dataframe.iloc[train_index]

        test_df = dataframe.iloc[test_index]
        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)

        # Model setup
        model = RobertaForSequenceClassification(
            RobertaConfig.from_pretrained('roberta-base', num_labels=len(mlb.classes_))
        )
        model.to(device)

        # Train and evaluate model across specified epochs
        train_and_evaluate(model, tokenizer, mlb, train_dataset, test_dataset, epochs)


def compare_label_distributions(label_counts, labels):
    results = []

    for i, fold_i in enumerate(label_counts):
        for j, fold_j in enumerate(label_counts):
            if i >= j:  # Avoid duplicate comparisons
                continue

            # Compare label distributions
            diff_train = {label: abs(fold_i['train'].get(label, 0) - fold_j['train'].get(label, 0)) for label in labels}
            diff_test = {label: abs(fold_i['test'].get(label, 0) - fold_j['test'].get(label, 0)) for label in labels}

            results.append({
                'Fold_1': fold_i['fold'],
                'Fold_2': fold_j['fold'],
                'Train_Diff': diff_train,
                'Test_Diff': diff_test,
            })

    # Convert to a DataFrame for easier visualization and saving
    comparison_df = pd.DataFrame(results)
    return comparison_df

def save_results_to_file(data, file_path, file_format='csv'):
    if file_format == 'csv':
        if isinstance(data, pd.DataFrame):
            data.to_csv(file_path, index=False)
        else:
            pd.DataFrame(data).to_csv(file_path, index=False)
    elif file_format == 'json':
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
    else:
        raise ValueError("Unsupported file format. Use 'csv' or 'json'.")


def plot_label_distribution(label_counts, labels):
    train_counts = {label: [] for label in labels}
    test_counts = {label: [] for label in labels}

    for count in label_counts:
        for label in labels:
            train_counts[label].append(count['train'].get(label, 0))
            test_counts[label].append(count['test'].get(label, 0))

    fig, ax = plt.subplots(figsize=(14, 8))
    for label in labels:
        ax.plot(range(1, len(label_counts) + 1), train_counts[label], label=f'Train {label}')
        ax.plot(range(1, len(label_counts) + 1), test_counts[label], label=f'Test {label}', linestyle='--')

    ax.set_xlabel('Fold Number')
    ax.set_ylabel('Label Count')
    ax.set_title('Label Distribution Across Folds')
    ax.legend()
    plt.show()

def load_dataset(path):
    dataframe = pd.read_csv(path)
    dataframe['context'] = dataframe['context'].fillna('')  # Replace NaN with empty string
    dataframe['context'] = dataframe['context'].astype(str)  # Ensure all entries are strings
    dataframe['DA_list'] = dataframe['DA'].apply(
        lambda x: list(sorted(set(x.replace(" ", "").split(',')))) if isinstance(x, str) else [])

    return dataframe


def main():
    if not os.path.exists("./logs"):
        os.mkdir("./logs")

    data_test = load_dataset('data/optimal_test.csv')
    data_train = load_dataset('data/optimal_train.csv')

    mlb = MultiLabelBinarizer()
    mlb.fit(data_train['DA_list'])
    mlb.fit(data_test['DA_list'])

    with open("models/mlb.pkl", "wb") as f:
        pickle.dump(mlb, f)

    dataset_train = Dataset.from_pandas(data_train)
    dataset_test = Dataset.from_pandas(data_test)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification(
        RobertaConfig.from_pretrained('roberta-base', num_labels=len(mlb.classes_))
    )
    model.to(device)

    train_and_evaluate(model, tokenizer, mlb, dataset_train, dataset_test, 30)


    # Execute
    # k_fold_cross_validation(dataframe, tokenizer, k=5, epochs=10)


if __name__ == '__main__':
    main()