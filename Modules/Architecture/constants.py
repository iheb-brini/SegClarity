from torch.cuda import is_available

DEFAULT_CHECKPOINT_PATH="models"

DEFAULT_LUNET_MODEL_PATH= f"{DEFAULT_CHECKPOINT_PATH}/lunet"
DEFAULT_UNET_MODEL_PATH= f"{DEFAULT_CHECKPOINT_PATH}/unet"


MODEL_TAGS=['from_scratch_models','finetuned_models','pretrained_models']
DEFAULT_DEVICE = "cuda:0" if is_available() else "cpu"
