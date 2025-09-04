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

image_path = './turbo_dpo_dataset/images'

with open('./turbo_dpo_dataset/data.json', 'r') as file:
    dataset = json.load(file)

conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models

def point_score(prompt, img):
    image_tensor = process_images([img], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

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
    return output


data = dataset[0]
image = Image.open(os.path.join(image_path, data['images'][0])).resize((512,512))
score = point_score(data['caption'], image)
print(f"Done running. Score: {score}")
