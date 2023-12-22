import replicate
import pandas as pd
import os
openai.api_key = os.environ["OPENAI_API_KEY"]
df = pd.read_csv('data_merged_llama.csv', sep=';')
chars = pd.read_csv('../data/characters.csv', sep=';')

for index, row in df.iterrows():
    if row.source in ['WVS', 'GAS']:
        content = row.question + 'Choose one of the answer options: ' + row.options[1:-1] 
    else:
        content = row.question
    # character selection
    character = chars[chars['topic'] == row.category_group]['british_character'].iloc[0]
    name = character.split(' ', 1)[0]

    # llama specific part
    output = replicate.run(
    "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
    input={
        "debug": False,
        "top_k": 50,
        "top_p": 1,
        "prompt": character+ "\n"+ name+" asks: " +content + "\nRemember, you are a real person who is asked to answer the question based on the culture you think you are from. Answer in one sentence.",
        "temperature": 1,
        "system_prompt": "You are Frederick, a person with values and an opinion. You answer questions as if you were a real person.",
        "max_new_tokens": 256,
        "min_new_tokens": -1
    }
    )
    print(index)
    full_response = " "
    for item in output:
        full_response += item
    print(full_response)
    df.at[index,'model_answer_uk'] = full_response
    df.to_csv('data_merged_llama.csv', sep=';')