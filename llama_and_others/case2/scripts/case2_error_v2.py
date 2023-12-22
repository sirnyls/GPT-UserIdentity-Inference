import pandas as pd 
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

df = pd.read_csv('essay_prompts_llama_7B_error_v2.csv', sep=';')

model_name_or_path = "TheBloke/Llama-2-7b-Chat-GPTQ"

model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
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
    print(tokenizer.decode(output[0]))
    df['essay_from_prompt_with_errors_v2'] = df['essay_from_prompt_with_errors_v2'].astype(str)
    df.at[index, 'essay_from_prompt_with_errors_v2'] = str(tokenizer.decode(output[0]))

    df.to_csv('essay_prompts_llama_7B_error_v2.csv', sep=';')


    