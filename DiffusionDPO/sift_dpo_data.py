from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
import random
import numpy as np
from PIL import Image
import requests
import copy
import torch
import tqdm
import sys
import warnings
import os
from datasets import load_dataset
import re
import json
from random import sample


warnings.filterwarnings("ignore")

model_name = "llava_qwen"
device = "cuda"
device_map = "auto"

# reward model
pretrained = "CodeGoat24/UnifiedReward-7b"
revision = "0958d1917a4c096b265c685e046ced33fcb09a7c"  # Before 15 Apr "Update". We need reproduction, MAN!
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, revision=revision)  # Add any other thing you want to pass in llava_model_args
model.eval()

save_path = './turbo_dpo_dataset/dpo_data.json'

image_path = './turbo_dpo_dataset/images'

save_image_path = './turbo_dpo_dataset/images_dpo/'

if not os.path.exists(save_image_path):
    os.makedirs(save_image_path)

with open('./turbo_dpo_dataset/data.json', 'r') as file:
    dataset = json.load(file)

conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models

def pair_rank(prompt, image1, image2):
    image_tensor = process_images([image1.resize((512, 512)), image2.resize((512, 512))], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

    conv_template = "qwen_1_5"

    question = f'<image>\n <image>\nYou are given a text caption and two generated images based on that caption. Your task is to evaluate and compare these images based on two key criteria:\n1. Alignment with the Caption: Assess how well each image aligns with the provided caption. Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n2. Overall Image Quality: Examine the visual quality of each image, including clarity, detail preservation, color accuracy, and overall aesthetic appeal.\nCompare both images using the above criteria and select the one that better aligns with the caption while exhibiting superior visual quality.\nProvide a clear conclusion such as \"Image 1 is better than Image 2.\", \"Image 2 is better than Image 1.\" and \"Both images are equally good.\"\nYour task is provided as follows:\nText Caption: [{prompt}]'

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [image1.size, image2.size]

    cont = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=4096
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
    output = text_outputs[0]

    if 'Image 2 is better than Image 1' in output:
        chosen = image2
        rejected = image1
    else:
        chosen = image1
        rejected = image2

    return chosen, rejected


def point_score(prompt, chosen_list, rejected_list):
    chosen_score = []
    rejected_score = []
    for img in chosen_list:
        image_tensor = process_images([img], image_processor, model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

        conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models

        question = f'<image>\nYou are given a text caption and a generated image based on that caption. Your task is to evaluate this image based on two key criteria:\n1. Alignment with the Caption: Assess how well this image aligns with the provided caption. Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n2. Overall Image Quality: Examine the visual quality of this image, including clarity, detail preservation, color accuracy, and overall aesthetic appeal.\nExtract key elements from the provided text caption, evaluate their presence in the generated image using the format: \'element (type): value\' (where value=0 means not generated, and value=1 means generated), and assign a score from 1 to 10 after \'Final Score:\'.\nYour task is provided as follows:\nText Caption: [{prompt}]'

        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        image_sizes = [img.size]

        cont = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=4096
        )
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
        output = text_outputs[0].split('Final Score:')[-1].strip()

        import re
        output = re.findall(r"\d+",output)
        if len(output) > 0:
            output = output[0]
        else:
            output = 0

        chosen_score.append(output)

    for img in rejected_list:
        image_tensor = process_images([img], image_processor, model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

        conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models

        question = f'<image>\nYou are given a text caption and a generated image based on that caption. Your task is to evaluate this image based on two key criteria:\n1. Alignment with the Caption: Assess how well this image aligns with the provided caption. Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n2. Overall Image Quality: Examine the visual quality of this image, including clarity, detail preservation, color accuracy, and overall aesthetic appeal.\nExtract key elements from the provided text caption, evaluate their presence in the generated image using the format: \'element (type): value\' (where value=0 means not generated, and value=1 means generated), and assign a score from 1 to 10 after \'Final Score:\'.\nYour task is provided as follows:\nText Caption: [{prompt}]'

        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        image_sizes = [img.size]

        cont = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=4096
        )
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
        output = text_outputs[0].split('Final Score:')[-1].strip()

        import re
        output = re.findall(r"\d+",output)
        if len(output) > 0:
            output = output[0]
        else:
            output = 0

        rejected_score.append(output)

    chosen = chosen_list[chosen_score.index(max(chosen_score))]
    rejected = rejected_list[rejected_score.index(min(rejected_score))]

    chos_score = max(chosen_score)
    rej_score = min(rejected_score)

    return chosen, rejected, chos_score, rej_score


if os.path.exists(save_path):
    with open(save_path, 'r') as file:
        data_list = json.load(file)
else:
    data_list = []

for i in tqdm.trange(len(dataset)):
    if i < len(data_list):
        continue
    data = dataset[i]
    prompt = data['caption']

    chosen_list = []
    rejected_list = []
    for j in range(5):

        image1 = Image.open(os.path.join(image_path, data['images'][2*j])).resize((512, 512))
        image2 = Image.open(os.path.join(image_path, data['images'][2*j+1])).resize((512, 512))

        chosen, rejected = pair_rank(prompt, image1, image2)

        chosen_list.append(chosen)
        rejected_list.append(rejected)


    chosen, rejected, chosen_score, rejected_score = point_score(prompt, chosen_list, rejected_list)

    if int(chosen_score) <= int(rejected_score):
        continue

    chosen.save(os.path.join(save_image_path, f"image_{i}_chosen.png"))
    rejected.save(os.path.join(save_image_path, f"image_{i}_rejected.png"))


    data['jpg_0'] = f'image_{i}_chosen.png'
    data['jpg_1'] = f'image_{i}_rejected.png'
    data['label_0'] = 1
    data['chosen_score'] = chosen_score
    data['rejected_score'] = rejected_score

    del data['images']

    data_list.append(data)

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)
