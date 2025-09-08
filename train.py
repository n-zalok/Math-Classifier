from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5TokenizerFast, DataCollatorForSeq2Seq
import random
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import numpy as np
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
    def __init__(self, tokenizer, model, p_irrelevant=0.05, p_relevant=0.20, max_source_length=512, max_target_length=512):
        self.tokenizer = tokenizer
        self.model = model
        self.base_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        self.p_irrelevant = p_irrelevant
        self.p_relevant = p_relevant
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __call__(self, features):
        new_features = []
        texts_pool = [f["text"] for f in features]  # candidate irrelevant texts

        for f in features:
            title = f["title"]
            sub_title = f["sub_title"]
            if not title:
                title = ""
            elif not sub_title:
                sub_title = ""
            else:
                pass
            context_relevant = f["text"]  # the real one
            # pick another sample’s text as irrelevant
            if len(features) > 1:
                context_irrelevant = random.choice(texts_pool)
                while context_irrelevant == context_relevant:
                    context_irrelevant = random.choice(texts_pool)
            else:
                context_irrelevant = ""

            r = random.random()
            if r < self.p_irrelevant:
                input_text = f"EXPLAIN {sub_title} {title} CONTEXT {context_irrelevant}"
            elif r < self.p_irrelevant + self.p_relevant:
                input_text = f"EXPLAIN {sub_title} {title} CONTEXT {context_relevant}"
            else:
                input_text = f"EXPLAIN {sub_title} {title}"

            label_text = f["text"]

            new_features.append({
                "input_ids": self.tokenizer(input_text, padding="max_length", truncation=True, max_length=self.max_source_length).input_ids,
                "labels": self.tokenizer(label_text, padding="max_length", truncation=True, max_length=self.max_target_length).input_ids
            })

        return self.base_collator(new_features)


custom_collator = CustomDataCollator(tokenizer, model, p_irrelevant=0.05, p_relevant=0.20, max_source_length=512, max_target_length=340)

training_args = Seq2SeqTrainingArguments(
    output_dir="t5_explain_runs/exp3",
    remove_unused_columns=False,
    per_device_train_batch_size=4,          # adjust to your VRAM
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=16,          # → effective batch 64
    num_train_epochs=3,
    eval_strategy="steps",
    eval_steps=1000,
    logging_steps=100,
    predict_with_generate=True,
    generation_max_length=340,
    generation_num_beams=4,
    label_smoothing_factor=0.05,
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