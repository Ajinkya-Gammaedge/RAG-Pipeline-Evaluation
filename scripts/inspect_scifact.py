from beir.datasets.data_loader import GenericDataLoader

data_path = "data/scifact"

corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

print(len(corpus), len(queries), len(qrels))
