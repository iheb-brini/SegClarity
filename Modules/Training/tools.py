from Modules.Dataset.constants import UTP_LABELS,SPLITAB1_LABELS

import numpy as np

def visualize(V,dataset_type,label_colors=None):
    if label_colors is None:
        label_colors = UTP_LABELS if 'UTP' in dataset_type.upper() else SPLITAB1_LABELS
    label_colors = np.array(label_colors)
    r = V.copy()
    g = V.copy()
    b = V.copy()
    for l in range(len(label_colors)):
        r[V==l]=label_colors[l,0]
        g[V==l]=label_colors[l,1]
        b[V==l]=label_colors[l,2]

    rgb = np.zeros((V.shape[0], V.shape[1], 3))
    rgb[:,:,0] = (r/255.0)
    rgb[:,:,1] = (g/255.0)
    rgb[:,:,2] = (b/255.0)
    return rgb