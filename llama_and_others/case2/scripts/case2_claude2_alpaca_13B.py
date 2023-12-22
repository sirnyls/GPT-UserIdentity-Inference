import pandas as pd 
import os
os.environ["TRANSFORMERS_CACHE"] = '/cluster/scratch/niheil/.cache/huggingface/transformers/'
os.environ["HF_HOME"] = '/cluster/scratch/niheil/.cache/huggingface'

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline




df = pd.read_csv('essay_prompts_claude2_alpaca_13B.csv', sep=';')

model_name_or_path = "umd-zhou-lab/claude2-alpaca-13B"

model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main", token = "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")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, token = "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")
for index, row in df.iterrows():
    prompt = row.prompt_with_errors_v2
    prompt_template=f'''[INST] <<SYS>>
    "Your task is to write an essay (about 300-350 words) in response to a question. The topic will be provided by the user.
    <</SYS>>
    {prompt}[/INST]

    '''
    print("\n\n*** Generate:")

    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, temperature=1, do_sample=True, top_p=1, top_k=40, max_new_tokens=512)
    print("Errors v2: ")
    print(index)
    print(tokenizer.decode(output[0]))
    df['essay_from_prompt_with_errors_v2'] = df['essay_from_prompt_with_errors_v2'].astype(str)
    df.at[index, 'essay_from_prompt_with_errors_v2'] = str(tokenizer.decode(output[0]))

    df.to_csv('essay_prompts_claude2_alpaca_13B.csv', sep=';')

df = pd.read_csv('essay_prompts_claude2_alpaca_13B.csv', sep=';')

for index, row in df.iterrows():
    prompt = row.essay_prompt_raw
    prompt_template=f'''[INST] <<SYS>>
    "Your task is to write an essay (about 300-350 words) in response to a question. The topic will be provided by the user.
    <</SYS>>
    {prompt}[/INST]

    '''
    print("\n\n*** Generate:")

    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, temperature=1, do_sample=True, top_p=1, top_k=40, max_new_tokens=512)
    print("Base: ")
    print(index)    
    print(tokenizer.decode(output[0]))
    df['essay_base'] = df['essay_base'].astype(str)
    df.at[index, 'essay_base'] = str(tokenizer.decode(output[0]))

    df.to_csv('essay_prompts_claude2_alpaca_13B.csv', sep=';')

df = pd.read_csv('essay_prompts_claude2_alpaca_13B.csv', sep=';')

for index, row in df.iterrows():
    prompt = row.essay_prompt_AAE
    prompt_template=f'''[INST] <<SYS>>
    "Your task is to write an essay (about 300-350 words) in response to a question. The topic will be provided by the user.
    <</SYS>>
    {prompt}[/INST]

    '''
    print("\n\n*** Generate:")

    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, temperature=1, do_sample=True, top_p=1, top_k=40, max_new_tokens=512)
    print("AAE: ")
    print(index)      
    print(tokenizer.decode(output[0]))
    df['essay_from_AAE_prompt'] = df['essay_from_AAE_prompt'].astype(str)
    df.at[index, 'essay_from_AAE_prompt'] = str(tokenizer.decode(output[0]))

    df.to_csv('essay_prompts_claude2_alpaca_13B.csv', sep=';')

df = pd.read_csv('essay_prompts_claude2_alpaca_13B.csv', sep=';')

for index, row in df.iterrows():
    prompt = row.prompt_with_errors
    prompt_template=f'''[INST] <<SYS>>
    "Your task is to write an essay (about 300-350 words) in response to a question. The topic will be provided by the user.
    <</SYS>>
    {prompt}[/INST]

    '''
    print("\n\n*** Generate:")

    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, temperature=1, do_sample=True, top_p=1, top_k=40, max_new_tokens=512)
    print("Error v1: ")
    print(index)   
    print(tokenizer.decode(output[0]))
    df['essay_from_error_prompt'] = df['essay_from_error_prompt'].astype(str)
    df.at[index, 'essay_from_error_prompt'] = str(tokenizer.decode(output[0]))

    df.to_csv('essay_prompts_claude2_alpaca_13B.csv', sep=';')