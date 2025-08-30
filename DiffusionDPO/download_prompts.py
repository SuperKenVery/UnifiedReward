from datasets import load_dataset
import json

ds = load_dataset("yuvalkirstain/PickaPic-selected-prompts")
train = ds['train']

prompts = list(train)
prompts_json = json.dumps(prompts)

with open("gen_data_prompts.json", "w") as f:
    f.write(prompts_json)
