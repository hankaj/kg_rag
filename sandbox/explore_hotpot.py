from datasets import load_dataset

cache_dir = "/Users/hania/school/magisterka/kg_rag/src/data/datasets"

dataset = load_dataset("hotpot_qa", 'distractor', cache_dir=cache_dir, trust_remote_code=True)
data = dataset['train']
print(data[4])