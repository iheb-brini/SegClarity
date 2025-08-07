SPLITAB1_BASE_DIR="/Users/rahaln/Desktop/Bureau/Datasets/divadb-960x1344/splitAB1"
UTP_BASE_DIR="/Users/rahaln/Desktop/New_datasets/Zone_labeling/UTP_Last_version_data_experiements_10-10-2023/30_UTP_from_296_UTP"

IMG_TAG = 'codex'
MASK_TAG = 'truthL'

# labels
SPLITAB1_BACKGROUND = [0, 0, 0]  # black
SPLITAB1_TEXT = [255, 255, 0]  # yellow
SPLITAB1_HIGHLITGH = [255, 0, 255]  # magenta
SPLITAB1_GLOSSES = [0, 255, 255]  # cyan


SPLITAB1_LABELS=[
    SPLITAB1_BACKGROUND,
    SPLITAB1_TEXT,
    SPLITAB1_HIGHLITGH,
    SPLITAB1_GLOSSES
]


# labels
UTP_BACKGROUND = [0, 0, 0]  # BLACK
UTP_DECOR = [255, 0, 0]  # RED
UTP_FILLER  = [0, 255, 0]  # GREEN
UTP_LINE = [0, 0, 255]  # BLUE
UTP_ZLINE = [255, 255, 0] # YELLOW
UTP_SINIT = [255, 0, 255] # MAGENTA
UTP_BINIT = [0, 255, 255] # CYAN

UTP_LABELS= [
    UTP_BACKGROUND,
    UTP_DECOR,
    UTP_FILLER,
    UTP_LINE,
    UTP_ZLINE,
    UTP_SINIT,
    UTP_BINIT
]

OUT_CHANNELS_PER_DATASET = {
     'splitAB1':4,
     'UTP':7
}