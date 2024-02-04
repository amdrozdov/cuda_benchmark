import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def draw(path, outname, title):
    """
    Basic bar chat visualisation for CPU vs GPU benchmark
    """
    ds = pd.read_csv(path)
    x = list(ds['Size'])
    cpu = list(ds['CPU'])
    gpu = list(ds['GPU'])

    max_y = max(max(cpu), max(gpu))
    benches = {'gpu': [], 'cpu': []}
    captions = []
    for i, v in enumerate(x):
        benches['cpu'].append(cpu[i])
        benches['gpu'].append(gpu[i])
        captions.append(str(v))

    fig, ax = plt.subplots(layout='constrained')
    width = 0.25
    multiplier = 0
    x = np.arange(len(captions))
    for attribute, measurement in benches.items():
        offset = width * multiplier
        rects = ax.bar(
            x + offset, measurement,
            width,log=True,ec="k", label=attribute
        )
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel('mk sec')
    ax.set_xlabel('Object size')
    ax.set_title(title)
    ax.set_xticks(x + width, captions)
    ax.legend(loc='upper left', ncols=3)

    ax.set_ylim(0, max_y)
    plt.savefig(outname, dpi=300)


if __name__ == '__main__':
    draw('add_benchmark.csv',
         'add.png', title='Vector add bench')
    draw('matrix_benchmark.csv',
         'matrix_mul.png', title='Matrix mul bench')
