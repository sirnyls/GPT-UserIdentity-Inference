import os
import openai
import pandas as pd
openai.api_key = os.environ["OPENAI_API_KEY"]
df = pd.read_csv('../data/data_merged.csv', sep=';')


for index, row in df[692:].iterrows():
    if row.source in ['WVS', 'GAS']:
        content = row.question + 'Choose one of the answer options: ' + row.options[1:-1] 
    else:
        content = row.question
    # character selection
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-1106",
    messages=[
        {
        "role": "system",
        "content": "You are Frederick, a person with values and an opinion. You answer questions as if you were a real person."
        },
        {
        "role": "user",
        "content":  "Julia asks: " +content + "\nRemember, you are a real person who is asked to answer the question. Answer in one sentence."
        }
    ],
    temperature=1,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    print(index)
    print(response['choices'][0]['message']['content'])
    df.at[index,'model_answer_neutral'] = response['choices'][0]['message']['content']
    df.to_csv('../data/data_merged.csv', sep=';')


