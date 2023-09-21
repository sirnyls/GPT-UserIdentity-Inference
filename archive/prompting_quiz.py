import os
import openai
import pandas as pd


quiz_original = pd.read_csv('data/usa_quiz.csv', encoding='utf-8')

prompt = "Please reformulate the question so that it is generic and can be answered for any country"
responses = [""]

for question in quiz_original.question:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
        temperature = 0.5
    )
    print(response['choices'][0]['message']['content'])
    responses.append(response['choices'][0]['message']['content'])

responses_df = pd.DataFrame(responses)

responses_df.to_csv('data/quiz_usa_questions_generic.csv', index=False)

