import re
import itertools
import sys
import time

from helpers import load_richardson_data, find_earliest_string, build_max_memory_mapping
from huggingface_hub import login
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

RATE_LIMIT = 0.5  # seconds after every OpenAI API call
ARROWS_OPT = {"text": ["up", "down", "left", "right"], "pseudo": ['↑', '↓', '←', '→']}
LLAMA_OPT = {
    "8b": "meta-llama/Meta-Llama-3.1-8B",
    "8b_inst": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "70b": "meta-llama/Meta-Llama-3.1-70B",
    "70b_inst": "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "70b_refl": "mattshumer/Reflection-Llama-3.1-70B",
    "r1-llama": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
}
GPT_OPT = {
    "4": "gpt-4o",
    "4-mini": "gpt-4o-mini",
    "3.5": "gpt-3.5-turbo",
    "o1-preview": "o1-preview"
}

GPT_API_KEY = ""
LLAMA_API_KEY = ""


def run_llama(prompt, model, tokenizer, max_length):
    tokens = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(tokens["input_ids"],
                            attention_mask=tokens["attention_mask"],
                            max_new_tokens=max_length,
                            num_return_sequences=1,
                            top_k=1)
    return tokenizer.decode(output[0], skip_special_tokens=False).strip()[len(prompt):]


def run_gpt(client, prompt, model_name, max_length):
    system_message = ("You are a participant in a research experiment. Even if the answer is subjective, provide it. "
                      "Do not say it is subjective. Follow the given structure.")

    if "o1" in MODEL:
        result = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": [{"type": "text", "text": system_message+' '+prompt}]}],
        )
    else:
        result = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": system_message},
                      {"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=max_length,
        )

    generated_answer = result.choices[0].message.content

    time.sleep(RATE_LIMIT)

    return generated_answer.strip()


def build_prompt(arrow_list, action_word):
    if GPT:
        if ANALOGY:
            return (
                f"EXAMPLE TASK: Given the concepts: 'X', '-', ')', '/'. For the concept that best represents the event 'stopped', what concept would you choose? Explain the analogy, then provide one choice."
                f"\nanalogy: 'stopping' often involves obstructing or halting the progress of something. Raising both arms and crossing them defensively to physically block someone for example."
                f"\nconcept: 'X'"
                f"\n\nTASK: Given the concepts: '{"', ".join(arrow_list)}'. For the concept that best represents the event '{action_word}', what concept would you choose? Explain the analogy, then provide one choice."
                f"\nanalogy:")
        elif ZEROSHOT:
            return f"TASK: Given the concepts: '{"', ".join(arrow_list)}'. For the concept that best represents the event '{action_word}', what concept would you choose? Give the chosen concept by surrounding it with '##'. Let's think step by step."
        else:
            return (
                f"EXAMPLE TASK: Given the concepts: 'X', '-', ')', '/'. For the concept that best represents the event 'stopped', what concept would you choose?"
                f"\nconcept: 'X'"
                f"\n\nTASK: Given the concepts: '{"', ".join(arrow_list)}'. For the concept that best represents the event '{action_word}', what concept would you choose?"
                f"\nconcept: ")
    else:
        if ANALOGY:
            return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

                You are a participant in a research experiment<|eot_id|><|start_header_id|>user<|end_header_id|>

                Given the concepts: 'X', '-', ')', '/'. For the concept that best represents the event 'stopped', what concept would you choose? Explain the analogy, then provide one choice.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

                analogy: 'stopping' often involves obstructing or halting the progress of something. Raising both arms and crossing them defensively to physically block someone for example.

                concept: 'X'<|eot_id|><|start_header_id|>user<|end_header_id|>

                Given the concepts: '{"', ".join(arrow_list)}'. For the concept that best represents the event '{action_word}', what concept would you choose? Explain the analogy, then provide one choice.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

                analogy: """
        elif ZEROSHOT:
            return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

                You are a participant in a research experiment<|eot_id|><|start_header_id|>user<|end_header_id|>

                Given the concepts: '{"', ".join(arrow_list)}'. For the concept that best represents the event '{action_word}', what concept would you choose? Give the chosen concept by surrounding it with '##'.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

                Let's think step by step."""
        else:
            return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

                You are a participant in a research experiment<|eot_id|><|start_header_id|>user<|end_header_id|>

                Given the concepts: 'X', '-', ')', '/'. For the concept that best represents the event 'stopped', what concept would you choose?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

                concept: 'X'<|eot_id|><|start_header_id|>user<|end_header_id|>

                Given the concepts: '{"', ".join(arrow_list)}'. For the concept that best represents the event '{action_word}', what concept would you choose?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

                concept: """


def run():
    start = time.time()
    model_choices = dict()
    arrows = ARROWS_OPT[ARROW]

    _, richardson_data, richardson_normed = load_richardson_data()
    action_words = list(richardson_normed.keys())

    if GPT:
        with open(GPT_API_KEY, 'r') as inf:
            api_key = inf.readline().strip()
            org_key = inf.readline().strip()
            proj_key = inf.readline().strip()

        model = None
        tokenizer = None
        client = OpenAI(api_key=api_key,
                        organization=org_key,
                        project=proj_key)
    else:
        with open(LLAMA_API_KEY) as file:
            access_token = file.readline().strip()
        login(token=access_token)
        tokenizer = AutoTokenizer.from_pretrained(LLAMA_OPT[MODEL])
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(LLAMA_OPT[MODEL],
                                                     device_map="auto",
                                                     max_memory=build_max_memory_mapping(),
                                                     trust_remote_code=True
                                                     )
        client = None

    print(f"Running Model: {MODEL} - Type: {ARROW} - Analogy: {ANALOGY}")
    for action_word in tqdm(action_words):
        for arrow_list in list(itertools.permutations(arrows)):
            prompt = build_prompt(arrow_list, action_word)
            num_tokens = 240 if ANALOGY else 800 if ZEROSHOT else 10
            generated_answer = run_gpt(client, prompt, GPT_OPT[MODEL], num_tokens) if GPT \
                else run_llama(prompt, model, tokenizer, num_tokens)

            if ANALOGY:
                match = re.search(r"[cC]oncept: .*", generated_answer)
                if match:
                    concept = match.group(0)
                else:
                    concept = ""
            elif ZEROSHOT:
                match = re.search(r"##.*##", generated_answer)
                if match:
                    concept = match.group(0)
                else:
                    concept = ""
            else:
                concept = generated_answer

            if action_word not in model_choices.keys():
                model_choices[action_word] = [0, 0, 0, 0]

            finding = find_earliest_string(arrows, concept)
            if finding:
                model_choices[action_word][arrows.index(finding)] += 1

            if sum(model_choices[action_word]) > 24:
                print("Too many arrows in response.")

            print(f"{action_word}\t{', '.join(arrow_list)}\t{generated_answer}\n----------",
                  file=open(GENERATED_ANSWER_FILE, "a+"))

    with open(CHOICES_FILE, "w+") as file:
        file.write(f"WORD\t{arrows[0]}\t{arrows[1]}\t{arrows[2]}\t{arrows[3]}\n")
        for key, values in model_choices.items():
            file.write(f"{key}\t{values[0]}\t{values[1]}\t{values[2]}\t{values[3]}\n")
        file.write(f"\nTotal time: {(time.time() - start) / 60} minutes")


if __name__ == "__main__":
    MODEL = sys.argv[1]
    if MODEL in GPT_OPT.keys():
        GPT = True
    elif MODEL in LLAMA_OPT.keys():
        GPT = False
    else:
        print("ERROR: Unrecognized model!")
        exit()

    ARROW = sys.argv[2]
    ANALOGY = sys.argv[3] == "1"
    ZEROSHOT = sys.argv[3] == "2"
    if ANALOGY is None or ARROW is None:
        print("Error.")
        exit()

    GENERATED_ANSWER_FILE = f"results/llms/{'zeroshot' if ZEROSHOT else "gpt" if GPT else "llama"}/generated_answers_{ARROW}_{MODEL}{"_analogy" if ANALOGY else ""}.txt"
    CHOICES_FILE = f"results/llms/{'zeroshot' if ZEROSHOT else "gpt" if GPT else "llama"}/choices_{ARROW}_{MODEL}{"_analogy" if ANALOGY else ""}.txt"
    run()
