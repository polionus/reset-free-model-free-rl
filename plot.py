import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import tyro
import glob


def main(log_dir: str = "logs/"):
    dfs = []
    for path in glob.glob(log_dir + "/*", recursive=True):
        df = pd.read_csv(path)
        df["cumsum"] = np.cumsum(df["reward"])
        dfs.append(df)

    df = pd.concat(dfs)

    sns.lineplot(data=df, x="step", y="cumsum", hue="reset free")
    plt.ylim(bottom=-10)
    plt.show()


if __name__ == "__main__":
    tyro.cli(main)
