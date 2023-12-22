import os
import openai
import pandas as pd

openai.api_key = os.environ["OPENAI_API_KEY"]
df = pd.read_csv('../data/sample.csv', sep=';')


for index, row in df.iterrows():
    response = openai.ChatCompletion.create(
    model="gpt-4-0613",
    messages=[
        {
        "role": "system",
        "content": "You are a helpful assistant for generating a realistic conversation history that aligns with a question that is provided."
        },
        {
        "role": "user",
        "content": """Generate a realistic conversation with 2 turns between a user and an AI assistant. Limit the conversation to max. 80 words. Satisfy the following requirements: 

                1. Generate the first version with 4 typical American vocabulary and the second version with everything else the same, but with the matching typical British words. Try to have the typical words in the user part, not the assistant part. Make the words bold.

                2. The conversation should be
                User (Turn 1): [your generated conversation]
                Assistant (Turn 1): [your generated conversation]
                User (Turn 2): [your generated conversation]
                Assistant (Turn 2): [your generated conversation]
                User (Turn 3): """+row.question+"""

                STOP.
                3. Make the conversation as realistic as possible. And ideally align with one of the top three use cases of GPT: writing assistance, educational tool or entertainment."""
        }
    ],
    temperature=1,
    max_tokens=1000,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    print(index)
    print(response['choices'][0]['message']['content'])
    df.at[index,'context'] = response['choices'][0]['message']['content']
    df.to_csv('../data/sample.csv', sep=';')


