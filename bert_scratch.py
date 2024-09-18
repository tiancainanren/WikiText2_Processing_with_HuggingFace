from transformers import BertForMaskedLM, BertTokenizer, Trainer, TrainingArguments, BertConfig
from datasets import load_dataset
import logging

# 配置logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logging.info("Starting BERT training from scratch...")

# 加载数据集
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
dataset = dataset["train"].train_test_split(test_size=0.1)
train_data = dataset["train"]
valid_data = dataset["test"]

# 定义tokenize函数
def tokenize_function(examples):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    encoding = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    encoding["labels"] = encoding["input_ids"].copy()
    return encoding

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_dataset = train_data.map(lambda examples: tokenize_function(examples), batched=True)
valid_dataset = valid_data.map(lambda examples: tokenize_function(examples), batched=True)

# 使用默认配置初始化BERT模型
bert_config = BertConfig()
bert_model = BertForMaskedLM(config=bert_config)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./bert_scratch",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    logging_dir="./logs_bert_scratch",
    logging_steps=10,
    save_strategy="no"
)

# 初始化Trainer
trainer = Trainer(
    model=bert_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset
)

# 开始训练
trainer.train()
logging.info("Evaluating BERT model trained from scratch...")
eval_result = trainer.evaluate()
logging.info(f"BERT model from scratch evaluation results: {eval_result}")
