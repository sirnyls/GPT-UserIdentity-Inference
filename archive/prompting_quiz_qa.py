import os
import openai
import pandas as pd
openai.api_key = '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'

q_a_a = pd.read_csv('data/quiz_questions_merged.csv', encoding='utf-8')

prompt = "You are an American person who should answer the provided question for the United States. Provide short, precise, truthful answers."
index = 0
for question in q_a_a.question:
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
        temperature=0
    )
    print(index)
    print(response['choices'][0]['message']['content'])
    q_a_a.answer_us[index] = response['choices'][0]['message']['content']
    index = index + 1
    q_a_a.to_csv('data/quiz_questions_merged.csv', index=False)

