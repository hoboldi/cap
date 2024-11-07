import numpy as np
import pandas as pd

anno_label_dict = pd.read_csv('data/annotation-label-dictionary.csv',
                              index_col='annotation', dtype='string')
print("Annotation-Label Dictionary")
print(anno_label_dict)

"Source: https://github.com/OxWearables/capture24"


def map_labels(data, anno_label_dict, label_mapping):
    data['label'] = (anno_label_dict[label_mapping]
                     .reindex(data['annotation'])
                     .to_numpy())
    return data


# Extract windows. Make a function as we will need it again later.
def extract_windows(data, winsize=10):
    X, Y = [], []
    for t, w in data.resample(f'{winsize}s', origin='start'):

        # Check window has no NaNs and is of correct length
        # 10s @ 100Hz = 1000 ticks
        if w.isna().any().any() or len(w) != winsize * 100:
            continue

        x = w[['x', 'y', 'z']].to_numpy()
        y = w['label'].mode(dropna=False).item()

        X.append(x)
        Y.append(y)

    X = np.stack(X)
    Y = np.stack(Y)

    return X, Y
