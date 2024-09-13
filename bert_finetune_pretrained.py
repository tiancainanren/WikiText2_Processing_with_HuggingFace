import logging
from transformers import BertForMaskedLM, BertTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# 配置logging，确保日志信息被写入文件
logging.basicConfig(
    filename='bert_finetune_pretrained_log.log',  # 日志文件名
    filemode='w',                 # 写入模式 'w' 表示覆盖，'a' 表示追加
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
    level=logging.INFO            # 设置日志级别为INFO
)

logging.info("Starting the training script...")

# 加载预训练的BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

# 加载数据集（使用wikitext-2数据集为例）
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# 对数据进行tokenize并添加labels
def tokenize_function(examples):
    # 将text编码为input_ids, attention_mask等
    encoding = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    encoding["labels"] = encoding["input_ids"].copy()  # 添加labels
    return encoding

logging.info("Tokenizing the dataset...")
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./bert_finetune_pretrained",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs_bert_finetune_pretrained",  # 日志保存目录
    logging_steps=10,  # 每10步记录一次日志
    save_steps=100,  # 每100步保存一次模型
    logging_strategy="steps",  # 记录日志的策略
)

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"]
)

# 开始训练
logging.info("Starting training...")
trainer.train()

# 保存模型
logging.info("Saving the trained model...")
model.save_pretrained("./bert_finetune_pretrained")
tokenizer.save_pretrained("./bert_finetune_pretrained")

logging.info("Training completed and model saved.")
