import os
import openai
import pandas as pd


q_a_a = pd.read_csv('data/anthropic_source_questions_WVS.csv', encoding='utf-8')

prompt = "Categorize the given question into one of the following topics: A. Social values and attitudes B. Religion and spirituality C. Science and technology D. Politics and policy E. Demographics F. Generations and age G. International affairs H. Internet and technology I. Gender and LGBTQ J. News habits and media K. Immigration and migration L. Family and relationships M. Race and ethnicity N. Economy and work O. Regions and countries P. Methodological research Q. Security"
index = 0
for question in q_a_a.question:
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Categorize the given question into ONLY ONE of the following topics: A. Social values and attitudes B. Religion and spirituality C. Science and technology D. Politics and policy E. Demographics F. Generations and age G. International affairs H. Internet and technology I. Gender and LGBTQ J. News habits and media K. Immigration and migration L. Family and relationships M. Race and ethnicity N. Economy and work O. Regions and countries P. Methodological research Q. Security"},
            {"role": "user", "content": question},
        ]
    )
    print(index)
    print(response['choices'][0]['message']['content'])
    q_a_a.category[index] = response['choices'][0]['message']['content']
    index = index + 1

q_a_a.to_csv('data/anthropic_source_questions_WVS.csv', index=False)

