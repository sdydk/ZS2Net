import matplotlib.pyplot as plt
import numpy as np
def decode_segmap(label_mask, dataset, plot=True):
    if dataset == "zooplankton":
        n_classes = 39
        label_colours = get_zooplankton_labels()
    else:
        raise NotImplementedError

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()

    for cls in range(0, n_classes):
        idx = label_mask == cls
        r[idx] = label_colours[cls, 0]
        g[idx] = label_colours[cls, 1]
        b[idx] = label_colours[cls, 2]
    rgb_image = np.stack([r, g, b], axis=2)
    return rgb_image
def get_zooplankton_labels():
    """
        background, Calanus sinicus Brodsky, Sagitta crassa Tokioka,  Themisto gracilipes, Penilia avirostris,
        Centropages abdominalis, Acartia pacifica, Centropages tenuiremis, Pontellopsis tenuicauda, Calanopia thompsoni,
        Sugiura chengshanense, Ophioplutues larva early,Eirene menoni, Euphausia pacifica, Evadne tergestina,
        Muggiaea atlantica, Paracalanus parvus, Oithona plumifera, Pleurobrachia globosa, Clytia folleata,
        Obelia dichotoma, Ectopleura bimanatus, Doliolum denticulatum, Oikopleura longicauda, Tornaria larva,
        Polychaeta larva early, Polychaeta larva later, Turritopsis nutricula, Proboscidactyla flavicirrata,
        Fritillaria formica, Labidocera rotunda, Alima larva, Megalopa larva, Brachyura zoea larva,
        Ophioplutues larva later, Fish eggs, Fish larva, Actinotrocha larva, Trochophora larva
    """
    return np.asarray(
        [
            [0, 0, 0],[128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128],
            [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
            [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128],
            [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [205, 91, 69],
            [47, 79, 79], [64, 224, 208], [75, 0, 130], [165, 42, 42], [102, 153, 255], [255, 127, 80], [0, 105, 139],
            [219, 112, 147], [255, 255, 224], [221, 160, 221], [0, 255, 255], [255, 215, 0]
        ]
    )