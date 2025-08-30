from diffusers import AutoPipelineForText2Image
import torch
import os
import random
import json

data_path = 'gen_data_prompts.json'
with open(data_path, 'r') as file:
    dataset = json.load(file)

pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    variant="fp16",
    low_cpu_mem_usage=False
)
pipe.to("cuda")
pipe.set_progress_bar_config(disable=True)

save_path = './turbo_dpo_dataset'
if not os.path.exists(save_path):
    os.makedirs(save_path)

save_image_path = os.path.join(save_path, 'images')
if not os.path.exists(save_image_path):
    os.makedirs(save_image_path)

save_file_path = os.path.join(save_path, 'data.json')


if os.path.exists(save_file_path):
    with open(save_file_path, 'r') as file:
        data_list = json.load(file)
else:
    data_list = []


import tqdm
print(f"Starting for loop!")
for i in tqdm.trange(len(dataset), desc="Generating images"):
    if i < len(data_list):
        continue
    data = dataset[i]

    prompt = data['prompt']
    seed = random.randint(0, 1000000)
    images = []

    # Create generators with different seeds for batch generation
    generators = [torch.Generator("cuda").manual_seed(seed+j) for j in range(10)]

    # Generate all 10 images in a single batch
    batch_images = pipe(
        prompt=[prompt] * 10,
        num_inference_steps=1,
        guidance_scale=0,
        generator=generators
    ).images

    # Save each image in the batch
    for j, image in enumerate(batch_images):
        image.save(os.path.join(save_image_path, f'image_{i}_{j}.png'))
        images.append(f"image_{i}_{j}.png")

    data['id'] = f'image_dpo_{i}'
    data['caption'] = data['prompt']
    data['images'] = images

    data_list.append(data)

    with open(save_file_path, 'w') as output_file:
        json.dump(data_list, output_file, indent=4)
