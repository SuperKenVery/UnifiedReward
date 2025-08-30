from datasets import load_dataset
import json, random

want_amount = 14_000
ds = load_dataset("yuvalkirstain/PickaPic-downloads")
train = ds['train']

prompts = list(train)
prompts = [
    {"prompt": data["prompt"]}
    for data in prompts
]
random.shuffle(prompts)
prompts_wanted = prompts[:want_amount]
prompts_json = json.dumps(prompts_wanted)

with open("gen_data_prompts.json", "w") as f:
    f.write(prompts_json)
