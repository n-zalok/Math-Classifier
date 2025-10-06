import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
from scipy.special import expit as sigmoid


ds = load_dataset("noor-zalouk/wiki-math-articles-multilabel")
print("Dataset loaded")


df = ds['train'].to_pandas()
all_labels = list(df['labels'].explode().unique())
mlb = MultiLabelBinarizer()
mlb.fit([all_labels])


model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
config = AutoConfig.from_pretrained(model_ckpt)

config.num_labels = len(all_labels)
config.id2label = {i: label for i, label in enumerate(all_labels)}
config.label2id = {label: i for i, label in enumerate(all_labels)}
config.problem_type = "multi_label_classification"

model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, config=config)
print("Model and tokenizer loaded")


def prepare(row): 
    inputs = tokenizer(row['input'], return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    label_ids = mlb.transform([row['labels']])[0]

    return {'input_ids': inputs['input_ids'].squeeze(), 'attention_mask': inputs['attention_mask'].squeeze(),
            'label_ids': torch.tensor(label_ids, dtype=torch.float)}

ds = ds.map(prepare)
ds = ds.remove_columns(['input', 'labels'])
print("Dataset prepared")

def compute_metrics(pred):
    y_true = pred.label_ids
    y_pred = sigmoid(pred.predictions)
    y_pred = (y_pred>0.5).astype(float)
    clf_report = classification_report(y_true, y_pred, target_names=mlb.classes_, zero_division=0, output_dict=True)
    return {"micro f1": clf_report["micro avg"]["f1-score"], "macro f1": clf_report["macro avg"]["f1-score"]}


training_args = TrainingArguments(
    output_dir="./BERT_multilabel", num_train_epochs=9, learning_rate=1e-5, lr_scheduler_type="constant",
    per_device_train_batch_size=64, per_device_eval_batch_size=64, gradient_accumulation_steps=1,
    warmup_ratio=0.1, eval_strategy="epoch", save_strategy="epoch", logging_strategy="epoch")


trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=ds["train"],
    eval_dataset=ds["valid"],
    processing_class=tokenizer)

trainer.train()