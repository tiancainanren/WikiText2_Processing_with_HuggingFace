from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, GPT2Config
from datasets import load_dataset
import logging

# 配置logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logging.info("Starting GPT-2 training from scratch...")

# 加载数据集
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
dataset = dataset["train"].train_test_split(test_size=0.1)
train_data = dataset["train"]
valid_data = dataset["test"]

# 定义tokenize函数
def tokenize_function(examples):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # 使用结束标记作为填充标记
    encoding = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    encoding["labels"] = encoding["input_ids"].copy()
    return encoding

gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token  # 添加填充标记
train_dataset = train_data.map(lambda examples: tokenize_function(examples), batched=True)
valid_dataset = valid_data.map(lambda examples: tokenize_function(examples), batched=True)

# 使用默认配置初始化GPT-2模型
gpt2_config = GPT2Config()
gpt2_model = GPT2LMHeadModel(config=gpt2_config)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./gpt2_scratch",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    logging_dir="./logs_gpt2_scratch",
    logging_steps=10,
    save_strategy="no"
)

# 初始化Trainer
trainer = Trainer(
    model=gpt2_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=gpt2_tokenizer  # 指定tokenizer
)

#
trainer.train()
logging.info("Evaluating GPT-2 model trained from scratch...")
eval_result = trainer.evaluate()
logging.info(f"GPT-2 model from scratch evaluation results: {eval_result}")