from torch.cuda import is_available

DEFAULT_DEVICE = "cuda:0" if is_available() else "cpu"