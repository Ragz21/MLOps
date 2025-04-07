import pandas as pd

def main():
    df = pd.read_csv("data/iris.csv")
    features = df.drop(columns=["target"])
    target = df["target"]
    features = (features - features.mean()) / features.std()

    df_processed = pd.concat([features, target], axis=1)

    df_processed.to_csv("data/iris_processed.csv", index=False)

if __name__ == "__main__":
    main()