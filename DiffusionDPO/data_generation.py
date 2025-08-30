from diffusers import AutoPipelineForText2Image
import torch
import os
import random
import json
import threading
import queue
import time
from PIL import Image

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

# Thread-safe queue for image saving
image_queue = queue.Queue()
stop_event = threading.Event()

# Worker function for saving images in a separate thread
def image_saver_worker():
    while not stop_event.is_set():
        try:
            # Get image data from queue with timeout
            image_data = image_queue.get(timeout=1)
            if image_data is None:  # Sentinel value to stop the worker
                break

            image, save_path = image_data
            image.save(save_path)
            image_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error saving image: {e}")
            image_queue.task_done()

# Start the image saver thread
saver_thread = threading.Thread(target=image_saver_worker, daemon=True)
saver_thread.start()

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

    # Queue each image for saving in the background thread
    image_paths = []
    for j, image in enumerate(batch_images):
        image_filename = f'image_{i}_{j}.png'
        image_path = os.path.join(save_image_path, image_filename)
        image_paths.append(image_filename)

        # Queue image for saving in background thread
        image_queue.put((image, image_path))

    data['id'] = f'image_dpo_{i}'
    data['caption'] = data['prompt']
    data['images'] = image_paths

    data_list.append(data)

    with open(save_file_path, 'w') as output_file:
        json.dump(data_list, output_file, indent=4)

# Wait for all images to be saved
image_queue.join()

# Stop the image saver thread
stop_event.set()
saver_thread.join()

print("All images saved successfully!")
