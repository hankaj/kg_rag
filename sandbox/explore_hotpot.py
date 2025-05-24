# from datasets import load_dataset

# cache_dir = "/Users/hania/school/magisterka/kg_rag/src/data/datasets"

# dataset = load_dataset("hotpot_qa", 'distractor', cache_dir=cache_dir, trust_remote_code=True)
# data = dataset['train']
# print(data[0])

import os

from groq import Groq

client = Groq(
    api_key='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',
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Explain the importance of fast language models",
        }
    ],
    model="llama-3.3-70b-versatile",
)

print(chat_completion.choices[0].message.content)