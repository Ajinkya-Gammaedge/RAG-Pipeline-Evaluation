from beir import util

dataset = "scifact"

url = (
    f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
)

data_path = util.download_and_unzip(url, "data")

print(f"SciFact downloaded to: {data_path}")
