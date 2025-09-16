import os
import torch
import torch.nn as nn
import torch.nn.functional as F
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


teacher_model_ckpt = "./BERT_multilabel/checkpoint-10572"
config = AutoConfig.from_pretrained(teacher_model_ckpt)
config.id2label = {i: label for i, label in enumerate(all_labels)}
config.label2id = {label: i for i, label in enumerate(all_labels)}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_model_ckpt, config=config).to(device)


student_ckpt = "distilbert-base-uncased"
student_config = AutoConfig.from_pretrained(student_ckpt, num_labels=config.num_labels,
                                            id2label=config.id2label, label2id=config.label2id)
student_tokenizer = AutoTokenizer.from_pretrained(student_ckpt)

def prepare(row):
    text = row['title']
    if row['sub_title']:
        text = text + ' ' + row['sub_title']
    else:
        pass

    text = text + ' ' + row['text']

    inputs = student_tokenizer(text, padding="max_length", truncation=True, max_length=512)
    label_ids = mlb.transform([row['category']])[0]

    inputs['label_ids'] = torch.tensor(label_ids, dtype=torch.float)

    return inputs

ds = ds.map(prepare)
ds = ds.remove_columns(['text', 'category', 'title', 'sub_title'])
print("Dataset prepared")


class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature

class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs_stu = model(**inputs)
        # Extract cross-entropy loss and logits from student
        loss_ce = outputs_stu.loss
        logits_stu = outputs_stu.logits
        # Extract logits from teacher
        with torch.no_grad():
            outputs_tea = self.teacher_model(**inputs)
            logits_tea = outputs_tea.logits
        # Soften probabilities and compute distillation loss
        loss_fct = nn.KLDivLoss(reduction="batchmean")
        loss_kd = self.args.temperature ** 2 * loss_fct(F.log_softmax(logits_stu / self.args.temperature, dim=-1),
                                                        F.softmax(logits_tea / self.args.temperature, dim=-1))
        # Return weighted student loss
        loss = (self.args.alpha * loss_ce) + ((1. - self.args.alpha) * loss_kd)
        return (loss, outputs_stu) if return_outputs else loss


def student_init():
    return (AutoModelForSequenceClassification.from_pretrained(
            student_ckpt, config=student_config).to(device))

def compute_metrics(pred):
    y_true = pred.label_ids
    y_pred = sigmoid(pred.predictions)
    y_pred = (y_pred>0.5).astype(float)
    clf_report = classification_report(y_true, y_pred, target_names=mlb.classes_, zero_division=0, output_dict=True)
    return {"micro f1": clf_report["micro avg"]["f1-score"], "macro f1": clf_report["macro avg"]["f1-score"]}


student_training_args = DistillationTrainingArguments(
    output_dir="./DistilBERT_multilabel", eval_strategy = "epoch",
    num_train_epochs=12, learning_rate=1e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=16,
    save_strategy="epoch", logging_steps=500,
    alpha=0.5, temperature=2, weight_decay=0.01)

distilbert_trainer = DistillationTrainer(
                     model_init=student_init,
                     teacher_model=teacher_model,
                     args=student_training_args,
                     train_dataset=ds['train'],
                     eval_dataset=ds['valid'],
                     compute_metrics=compute_metrics,
                     processing_class=student_tokenizer)

distilbert_trainer.train()