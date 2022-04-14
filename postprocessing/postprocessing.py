import os
import sys

import pandas as pd
from matplotlib import pyplot as plt


def ugp3_stats_plot(fname):
    basename = os.path.splitext(os.path.basename(fname))[0]
    dirname = os.path.dirname(fname)
    with open(fname, "r") as fin:
        df = pd.read_csv(fin)
        sub_df = df[["P1_Avg_f0", "P1_Best_f0", "P1_Worst_f0", "P1_Avg_f1", "P1_Best_f1", "P1_Worst_f1"]]
        sub_df.plot(secondary_y=["P1_Avg_f1", "P1_Best_f1", "P1_Worst_f1"], mark_right=False,
                    figsize=(max(8, int(0.33 * len(df))), 6))
        plt.xticks(range(len(df.index)), df.index)
        plt.savefig(os.path.join(dirname, basename))


def individuals_fitness_plot(fname, ugp3_stats):
    from matplotlib import lines
    _linestyles = list(lines.lineStyles.keys())

    basename = os.path.splitext(os.path.basename(fname))[0]
    dirname = os.path.dirname(fname)

    with open(ugp3_stats, "r") as fin:
        df = pd.read_csv(fin)
        individuals_per_generation = df["P1_EvalCount"].to_list()

    with open(fname, "r") as fin:
        df = pd.read_csv(fin, sep=" |\t", header=None, engine='python', names=["individual", "f0", "f1"],
                         index_col="individual")
        _sorter = sorted(df.index, key=lambda x: (x[:-1], x[-1].isdigit(), x))
        df = df.loc[_sorter]
        df.plot(x_compat=True, figsize=(max(8, int(0.33 * len(df))), 6))
        plt.xticks(range(len(df.index)), df.index)
        for _gen, xc in enumerate(individuals_per_generation):
            plt.vlines(x=xc, ymin=-10, ymax=100, colors='0.75', linestyles=_linestyles[_gen % len(_linestyles)],
                       label='generation ' + str(_gen))
        plt.legend()
        plt.savefig(os.path.join(dirname, basename))


if __name__ == "__main__":
    assert len(
        sys.argv) > 2, "Required args: <ugp3 stats file> <individuals fitness file>\nFigures will be output in <input_file_dir>/<input_file_name.png>"
    ugp3_stats_plot(sys.argv[1])
    individuals_fitness_plot(sys.argv[2], sys.argv[1])
