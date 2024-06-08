import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_NAME = "stabilityai/japanese-stablelm-2-base-1_6b"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    useFast=False,
    trust_remote_code=True,
)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"":0},
    trust_remote_code=True,
    torch_dtype="auto",
)
print(model)


# 学習結果をの保存先
from datetime import datetime
from pathlib import Path
session_path = Path(f'./session/{datetime.now().strftime("%Y%m%d%H%M%S")}_{MODEL_NAME.split("/")[-1]}')

from peft import prepare_model_for_kbit_training

model.enable_input_require_grads()
model.gradient_checkpointing_enable()

model = prepare_model_for_kbit_training(model)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

from peft import LoraConfig, get_peft_model

config = LoraConfig(
    #target_modules=["dense_4h_to_h", "dense", "dense_h_to_4h", "query_key_value"], 
    #target_modules=["query_key_value"],
    target_modules=[
    #    "embed_tokens",
    #    "lm_head",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    
    r=16, # 16, 32, 128
    lora_alpha=32,
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    inference_mode=False,
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

import datasets

prompt_template = """以下の文章を要約してください。

### 文章:
{input}

# 要約結果:
"""

def encode(sample):
    # summaryの文字数を取得し、１の位を切り上げる
    num = len(sample["summary"])
    num = -(-num // 10) * 10
    
    prompt = prompt_template.format(num=num, input=sample["input"])
    target = sample["summary"] + tokenizer.eos_token
    
    input_ids_prompt, input_ids_target = tokenizer([prompt, target]).input_ids
    input_ids = input_ids_prompt + input_ids_target
    
    labels = input_ids.copy()
    labels[:len(input_ids_prompt)] = [-100] * len(input_ids_prompt)
    
    return {"input_ids": input_ids, "labels": labels}

def get_collator(tokenizer, max_length):
    def collator(batch):
        batch = [{ key: value[:max_length] for key, value in sample.items() } for sample in batch ]
        batch = tokenizer.pad(batch, padding=True)
        batch["labels"] = [ e + [-100] * (len(batch["input_ids"][0]) - len(e)) for e in batch["labels"] ]
        batch = { key: torch.tensor(value) for key, value in batch.items() }
        return batch
    return collator

dataset = datasets.load_dataset('json', data_files='./dataset/three_line_summary.json')
dataset = dataset.map(encode)
dataset = dataset["train"].train_test_split(0.2)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

#  評価用のデータセットを保存しておく
eval_dataset.remove_columns(['input_ids', 'labels']).save_to_disk(f"./{session_path}/eval_dataset")

print(train_dataset)
print(tokenizer.decode(train_dataset[0]['input_ids'], skip_special_tokens=False))

import transformers
training_arguments = transformers.TrainingArguments(
    output_dir=f"./{session_path}/checkpoints",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    logging_strategy='steps',
    logging_steps=10,
    save_strategy='epoch',
    evaluation_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=2,
    #fp16=True,
    #optim="paged_adamw_32bit",
    optim="paged_adamw_8bit",
    neftune_noise_alpha=5, # NEFTuneのノイズの強さ
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_arguments,
    data_collator=get_collator(tokenizer, 2048)
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
#model.resize_token_embeddings(len(tokenizer))  # type: ignore

trainer.train()

model = trainer.model
peft_model_name = f"./{session_path}/model"
model.save_pretrained(peft_model_name)
