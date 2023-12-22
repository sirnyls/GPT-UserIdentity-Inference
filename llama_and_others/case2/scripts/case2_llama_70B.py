import pandas as pd 
import os
os.environ["TRANSFORMERS_CACHE"] = '/cluster/scratch/niheil/.cache/huggingface/transformers/'
os.environ["HF_HOME"] = '/cluster/scratch/niheil/.cache/huggingface'

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

df = pd.read_csv('essay_prompts_llama_70B.csv', sep=';')

model_name_or_path = "meta-llama/Llama-2-70b-chat-hf"

model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main", 
                                             token = "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")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, token = "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")


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
    print(index)
    print(tokenizer.decode(output[0]))
    df['essay_from_error_prompt'] = df['essay_from_error_prompt'].astype(str)
    df.at[index, 'essay_from_error_prompt'] = str(tokenizer.decode(output[0]))

    df.to_csv('essay_prompts_llama_70B.csv', sep=';')
