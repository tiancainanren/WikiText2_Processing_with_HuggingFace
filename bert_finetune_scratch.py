import logging
from transformers import BertConfig, BertForMaskedLM, Trainer, TrainingArguments,BertTokenizer
from datasets import load_dataset

# 配置logging
logging.basicConfig(
    filename='bert_finetune_scratch_log.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logging.info("Starting the BERT fine-tuning from scratch...")

# 加载数据集
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# 初始化BERT配置，不加载预训练模型权重
config = BertConfig()
model = BertForMaskedLM(config)

# 加载tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    encoding = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    encoding["labels"] = encoding["input_ids"].copy()
    return encoding

logging.info("Tokenizing dataset for BERT from scratch...")
tokenized_datasets = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./bert_finetune_scratch",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs_bert_finetune_scratch",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"]
)

logging.info("Starting training for BERT from scratch...")
trainer.train()

logging.info("Saving BERT from scratch model...")
model.save_pretrained("./bert_finetune_scratch")
tokenizer.save_pretrained("./bert_finetune_scratch")

logging.info("Training for BERT from scratch completed.")
