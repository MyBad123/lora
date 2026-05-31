import os
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments

# ================= 1. НАСТРОЙКИ МОДЕЛИ =================
# Читаем пути из переменных окружения. Если их нет, используем дефолтные.
MODEL_PATH = os.getenv("MODEL_PATH", "./my_model")
DATASET_PATH = os.getenv("DATASET_PATH", "dataset.jsonl")

max_seq_length = 2048
dtype = None
load_in_4bit = True

print(f"Загрузка модели из {MODEL_PATH}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_PATH,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# ================= 2. НАСТРОЙКА LORA =================
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# ================= 3. ПОДГОТОВКА ДАТАСЕТА =================
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "qwen-2.5",
)

def formatting_prompts_func(examples):
    texts = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in examples["messages"]]
    return {"text": texts}

print(f"Обработка датасета {DATASET_PATH} ...")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
dataset = dataset.map(formatting_prompts_func, batched = True)

# ================= 4. ОБУЧЕНИЕ =================
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        num_train_epochs = 3,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 5,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

print("Старт обучения 🚀")
trainer_stats = trainer.train()

# ================= 5. СОХРАНЕНИЕ ВЕСОВ =================
model.save_pretrained("lora_qwen_telegram")
tokenizer.save_pretrained("lora_qwen_telegram")
print("✅ Обучение завершено! LoRA веса сохранены в папку 'lora_qwen_telegram'.")