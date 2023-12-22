import os
import openai
import pandas as pd
openai.api_key = os.environ["OPENAI_API_KEY"]
df = pd.read_csv('new_sample_results.csv', sep=';')

df['similarity_gpt_uk'] = ''

for index, row in df[84:].iterrows():

    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
        "role": "system",
        "content": "Your task is to determine the similarity of two answers to a question."
        },
        {
        "role": "user",
        "content": """Output 1 if the answers go in the same direction with their meaning, else output 0.
                    Example:
                    Input:
                    question: Is it common to live with roommates? 
                    answer_1: Very common, especially in cities and during college.
                    answer_2: Yes, it is quite common to live with roommates, especially in cities like New York where housing can be expensive.
                    Output: 1
                    question: """+row.question+"""\nanswer1: """+row.answer_uk+
                    """\nanswer2: """+row.model_answer_uk
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
    df.at[index,'similarity_gpt_uk'] = response['choices'][0]['message']['content']
    df.to_csv('new_sample_results_gptscore_uk_84.csv', sep=';')


