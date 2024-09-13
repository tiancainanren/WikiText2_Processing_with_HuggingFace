import logging
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# 配置logging
logging.basicConfig(
    filename='gpt2_finetune_pretrained_log.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logging.info("Starting the GPT-2 fine-tuning with pretrained model...")

# 加载数据集
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# 加载预训练的GPT-2模型和tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# 对数据进行tokenize并添加labels
def tokenize_function(examples):
    encoding = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    encoding["labels"] = encoding["input_ids"].copy()
    return encoding

logging.info("Tokenizing dataset for GPT-2...")
tokenized_datasets = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./gpt2_finetune_pretrained",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs_gpt2_finetune_pretrained",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"]
)

logging.info("Starting training for GPT-2 fine-tuning...")
trainer.train()

logging.info("Saving GPT-2 fine-tuned model...")
model.save_pretrained("./gpt2_finetune_pretrained")
tokenizer.save_pretrained("./gpt2_finetune_pretrained")

logging.info("Training for GPT-2 fine-tuning completed.")
