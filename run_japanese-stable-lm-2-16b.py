import torch
import peft
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time

# ベースモデルとLoRAモデルを選択
MODEL_NAME = "stabilityai/japanese-stablelm-2-base-1_6b"
peft_model_path = "./session/20240531123758_japanese-stablelm-2-base-1_6b"
#MODEL_NAME = "./merged/20240527114505_japanese-stablelm-2-base-1_6b"

print(f"### {MODEL_NAME} ###")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    padding_side="left",
    trust_remote_code=True,
    )

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"":0},
    trust_remote_code=True,
    )
model = peft.PeftModel.from_pretrained(model, f"{peft_model_path}/model")

# プロンプトのテンプレート
prompt_template = """以下の文章を要約してください。

### 文章:
{input}

# 要約結果:
"""

# 評価用データセットを読み込む
from datasets import load_from_disk
eval_datasets = load_from_disk(f"./{peft_model_path}/eval_dataset")

# ランダムな5件のみ使用
eval_datasets = eval_datasets.shuffle(seed=42).select(range(5))

# 推論
start_time = time.time()

for eval_data in eval_datasets:
    prompt = prompt_template.format(input=eval_data["input"])
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        tokens = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            #do_sample=True,
            #temperature=0.5,
            #top_p=0.75, 
            #top_k=20,         
            #repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    output = tokenizer.decode(tokens[0], skip_special_tokens=False)
    output = output.split(f"# 要約結果:\n")[1].replace(tokenizer.eos_token, "")
    
    print("### 入力 ###")
    print(eval_data["input"])
    print("")
    print("### 出力 ###")
    print(output)
    print("---")

end_time = time.time()
elapsed_time = end_time - start_time

#print(input_text)
print("Elapsed time:", elapsed_time, "seconds")
