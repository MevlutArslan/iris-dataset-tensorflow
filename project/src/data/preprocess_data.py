import pandas as pd
from pathlib import Path
import sys

path = Path(__file__).parent / "../../data/iris_csv.csv"
df = pd.read_csv(path)

y = df["class"]

class_map = {"Iris-setosa": 0.1, "Iris-versicolor": 0.2, "Iris-virginica": 0.3}

y = y.map(class_map)

df.pop("class")

sepallength_max = df["sepallength"].max()

df.sepallength = df.sepallength.apply(lambda num: num / sepallength_max)

sepalwidth_max = df["sepalwidth"].max()

df.sepalwidth = df.sepalwidth.apply(lambda num: num / sepalwidth_max)

petallength_max = df["petallength"].max()

df.petallength = df.petallength.apply(lambda num: num / petallength_max)

petalwidth_max = df["petalwidth"].max()

df.petalwidth = df.petalwidth.apply(lambda num: num / petalwidth_max)

# print(df.shape)
x_train = df[:120]
y_train = y[:120]

x_validate = df[120:]
y_validate = y[120:]

x_dim = len(df.iloc[0])