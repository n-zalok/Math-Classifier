from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5TokenizerFast, DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"



ds = load_dataset("noor-zalouk/wiki-math-articles")
print("Dataset loaded")

model_name = "t5-small"
tokenizer = T5TokenizerFast.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
base_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
print("Model, Tokenizer and Collator loaded")

class CustomDataCollator:
    def __init__(self, tokenizer, model, max_source_length=512, max_target_length=250):
        self.tokenizer = tokenizer
        self.model = model
        self.base_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __call__(self, features):
        new_features = []

        for f in features:
            title = f["title"]
            sub_title = f["sub_title"]
            if not title:
                title = ""
            elif not sub_title:
                sub_title = ""
            else:
                pass

            input_text = f"EXPLAIN {sub_title} {title}"

            label_text = f["text"]

            new_features.append({
                "input_ids": self.tokenizer(input_text, padding="max_length", truncation=True, max_length=self.max_source_length).input_ids,
                "labels": self.tokenizer(label_text, padding="max_length", truncation=True, max_length=self.max_target_length).input_ids
            })

        return self.base_collator(new_features)


custom_collator = CustomDataCollator(tokenizer, model, max_source_length=512, max_target_length=250)

training_args = Seq2SeqTrainingArguments(
    output_dir="t5_explain_runs/exp4",
    remove_unused_columns=False,
    per_device_train_batch_size=4,          # adjust to your VRAM
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=16,          # → effective batch 64
    num_train_epochs=4,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
    label_smoothing_factor=0.1,
    warmup_ratio=0.1,
    learning_rate=1e-4,
    weight_decay=0.01,
    seed=42,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=ds['train'],             # your tokenized dataset objects
    eval_dataset=ds['valid'],
    data_collator=custom_collator,
    processing_class=tokenizer
)

trainer.train()