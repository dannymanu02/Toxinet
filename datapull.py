import pandas as pd

df = pd.read_csv("hf://datasets/TLeonidas/twitter-hate-speech-en-240ksamples/recombined_data.csv")

df.to_csv("hate_speech.csv", header=True, index=False)