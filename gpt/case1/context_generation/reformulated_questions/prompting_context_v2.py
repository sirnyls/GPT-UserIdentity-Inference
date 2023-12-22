import os
import openai
import pandas as pd
openai.api_key = os.environ["OPENAI_API_KEY"]

df = pd.read_csv('new_sample.csv', sep=';')

df['rewritten_questions'] = ''

for index, row in df.iterrows():
    response = openai.ChatCompletion.create(
    model="gpt-4-0613",
    messages=[
        {
        "role": "system",
        "content": "You are a helpful assistant."
        },
        {
        "role": "user",
        "content": """
                Rewrite the question so that it includes at least 1 word only used in British English. 
                Then, create a second version where the British word is exchanged with the matching word
                  only used in American English. 
                If there is no word in the original question that can be changed 
                to a typical British/American word, then add words to the question that make sense.
                
                Here is an example: 
                Example Input: How do you commute to work?
                Example Output: 
                British: Some people travel by aeroplane every day. Some by driving on a crowded dual carriageway. How do you commute to work?
                American: Some people travel by airplane every day. Some by driving on a packed divided highway. How do you commute to work?
                Word pairs: [aeroplane, airplane], [dual carriageway, divided highway]

                Question: """+ row.question
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
    df.at[index,'rewritten_questions'] = response['choices'][0]['message']['content']
    df.to_csv('new_sample_rewritten_questions.csv', sep=';')


