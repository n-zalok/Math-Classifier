import os
import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
from scipy.special import expit as sigmoid


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"


ds = load_dataset("noor-zalouk/wiki-math-articles-multilabel")
print("Dataset loaded")


df = ds['test'].to_pandas()
all_labels = list(df['category'].explode().unique())
mlb = MultiLabelBinarizer()
mlb.fit([all_labels])


model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
config = AutoConfig.from_pretrained(model_ckpt)
config.num_labels = len(all_labels)
config.problem_type = "multi_label_classification"
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, config=config)


def prepare(row):
    text = row['title']
    if row['sub_title']:
        text = text + ' ' + row['sub_title']
    else:
        pass

    text = text + ' ' + row['text']

    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=512)
    label_ids = mlb.transform([row['category']])[0]

    inputs['label_ids'] = torch.tensor(label_ids, dtype=torch.float)

    return inputs

ds = ds.map(prepare)
ds = ds.remove_columns(['text', 'category', 'title', 'sub_title'])
print("Dataset prepared")


def compute_metrics(pred):
    y_true = pred.label_ids
    y_pred = sigmoid(pred.predictions)
    y_pred = (y_pred>0.5).astype(float)
    clf_report = classification_report(y_true, y_pred, target_names=mlb.classes_, zero_division=0, output_dict=True)
    return {"micro f1": clf_report["micro avg"]["f1-score"], "macro f1": clf_report["macro avg"]["f1-score"]}


training_args = TrainingArguments(
    output_dir="./BERT_multilabel", num_train_epochs=12, learning_rate=1e-5,
    per_device_train_batch_size=4, per_device_eval_batch_size=4, gradient_accumulation_steps=16,
    weight_decay=0.01, warmup_ratio=0.1, eval_strategy="epoch", save_strategy="epoch", logging_steps=100)


trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=ds["train"],
    eval_dataset=ds["valid"],
    processing_class=tokenizer)

trainer.train()