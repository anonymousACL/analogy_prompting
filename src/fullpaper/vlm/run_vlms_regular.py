import tqdm 
from tqdm.auto import tqdm as base_tqdm
print(tqdm.__version__)
from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig, CLIPProcessor, CLIPModel, Qwen2VLForConditionalGeneration, AutoTokenizer
import torch
from PIL import Image
import requests
import pandas as pd
from utils.paths import IMG_DIR, ROOT
from utils.utils import *
from utils.supported_vlms import hf_model_ids
import argparse
import json
from itertools import permutations

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', choices=['molmo', 'qwen-vl', 'clip', 'qwen-vl-72b', 'molmo-72b'])

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_id = hf_model_ids[args.model]
max_new_tokens = 20


temperature = 0

# Importing images
up = Image.open(IMG_DIR / 'up.png')
down = Image.open(IMG_DIR / 'down.png')
left = Image.open(IMG_DIR / 'left.png')
right = Image.open(IMG_DIR / 'right.png')

x = Image.open(ROOT / 'data/images/example_images/x.png')
bracket = Image.open(ROOT / 'data/images/example_images/).png')
dash = Image.open(ROOT / 'data/images/example_images/-.png')
slash = Image.open(ROOT / 'data/images/example_images/:.png')

schema_choice = Image.open(IMG_DIR / 'schema_choice_updated.png')


img_labels = ['SHTL', 'XHWK', 'AKRC', 'ZHRN']
label_permutations = list(permutations(img_labels))
label_to_image = {img_label: image for img_label, image in zip(img_labels, [left, right, up, down])}

_, richardson_data, richardson_normed = load_richardson_data()
action_words = list(richardson_normed.keys())


out_dict = {}

if args.model == 'qwen-vl' or args.model == 'qwen-vl-72b':
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto", device_map='auto')
    processor = AutoProcessor.from_pretrained(model_id)
    print('Model loaded')

    for action_word in action_words:
        print(action_word)
        out_dict[action_word] = {}
        for label_permutation in label_permutations:
            label_1, label_2, label_3, label_4 = label_permutation

            conversation = [
                {

                "role": "user",
                "content": [

                    {"type": "text", "text": "Example task: Consider the event 'stopped' and these four images: SHTL "},
                    {"type": "image"},
                    {"type": "text", "text": "XHWK "},
                    {"type": "image"},
                    {"type": "text", "text": "AKRC "},
                    {"type": "image"},
                    {"type": "text", "text": "ZHRN "},
                    {"type": "image"},
                    {"type": "text", "text": f"Which of the images best represents the event?\nImage: SHTL\n\nTask: Consider the event '{action_word}' and these four images: {label_1} "},
                    {"type": "image"},
                    {"type": "text", "text": f"{label_2} "},
                    {"type": "image"},
                    {"type": "text", "text": f"{label_3} "},
                    {"type": "image"},
                    {"type": "text", "text": f"{label_4} "},
                    {"type": "image"},
                    {"type": "text", "text": f"Which of the images best represents the event?\nImage: "},
                    ]
                },
            ]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(images=[x, dash, bracket, slash]+[label_to_image[label_1], label_to_image[label_2], label_to_image[label_3], label_to_image[label_4]], 
                               text=prompt, padding=True, return_tensors='pt').to(device)
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature = temperature)
            generated_ids = [
                    output_ids[len(input_ids) :]
                    for input_ids, output_ids in zip(inputs.input_ids, output_ids)
                ]

            
            decoded_output = processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            out_dict[action_word]['_'.join(label_permutation)] = decoded_output
            del inputs, output_ids, generated_ids, decoded_output
            # break


elif args.model == 'molmo' or args.model=='molmo-72b':

    processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
    )


    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )
    print('Model loaded')
    print(torch.cuda.device_count())
    print(model.device)
    for action_word in action_words:

        print(action_word)
        
        out_dict[action_word] = {}
         
        for label_permutation in label_permutations:
            
            label_1, label_2, label_3, label_4 = label_permutation
            prompt = f"Example task: Consider the event 'threw' and the four images below (SHTL, XHWK, AKRC, ZHRN). Which of the images best represents the event?\nImage: XHWK\n\nTask: Consider the event '{action_word}' and the four images below (SHTL, XHWK, AKRC, ZHRN). Which of the images best represents the event?\nImage: "
            
            inputs = processor.process(
                images=[schema_choice],
                text=prompt
            )

            inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

            output = model.generate_from_batch(
                    inputs,
                    GenerationConfig(max_new_tokens=max_new_tokens, stop_strings="<|endoftext|>"), tokenizer=processor.tokenizer, temperature = temperature
                    )
            generated_tokens = output[0,inputs['input_ids'].size(1):]
            decoded_output = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            out_dict[action_word]['_'.join(label_permutation)] = decoded_output
            del inputs, output, generated_tokens, decoded_output

            
elif args.model == 'clip':

    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)


    for action_word in action_words:
        print(action_word)
        inputs = processor(text=action_word, images=[up, down, left, right], return_tensors="pt", padding=True).to(device)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image.flatten().softmax(dim=0).detach().cpu().numpy().tolist()
        print(logits_per_image)
        word_dict = {w:logit for w,logit in zip(['up', 'down', 'left', 'right'], logits_per_image)}
        out_dict[action_word] = word_dict
        # break

    

    
dict_to_json(out_dict, ROOT / f'src/fullpaper/results/vlms/{args.model}_human_oneshot.json')



