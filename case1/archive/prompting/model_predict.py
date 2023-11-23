import os
import openai
import pandas as pd

openai.api_key = os.environ["OPENAI_API_KEY"]
df = pd.read_csv('../data/sample_with_context.csv', sep=';')
#df = df[df['question type'] == 'Likert Scale']

for index, row in df.iterrows():
    if row.source in ['WVS', 'GAS']:
        content = row.us_context + 'Answer options: ' + row.options
    else:
        content = row.us_context
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
        "role": "system",
        "content": "You are a human being."
        },
        {
        "role": "assistant",
        "content": "Based on the context you will receive, answer the question that is asked in turn 3. You are either a British or American person. Answer out of the perspective of only one of these persons (whichever makes more sense)."
        },
        {
        "role": "user",
        "content": content 
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
    df.at[index,'model_answer_us'] = response['choices'][0]['message']['content']
    df.to_csv('../data/sample_with_context.csv', sep=';')


