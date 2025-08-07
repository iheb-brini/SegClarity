from Modules.Architecture import generate_lunet_model,generate_unet_model
from Modules.ModelXAI import generate_XAI_model
from Modules.Attribution import generateAttributions
from Modules.Utils import Normalizations
from Modules.Attribution import LUNET_LAYERS as layers_dict
from .constants import DEFAULT_DEVICE
import os
import torch


class BaseExperiment:
    def __init__(self, out_channels=4,device=DEFAULT_DEVICE, model_type = 'lunet',checkpoint_name=None, models_path=None,ld=None,model=None) -> None:
        self.out_channels = out_channels
        self.device = device
        self.model_type = model_type
        self.methods = ['LayerLRP', 'LayerGradCam',
                        'LayerDeepLift', 'LayerGradientXActivation']
        self.layers_dict = ld if ld is not None else layers_dict
        if checkpoint_name is not None and models_path is not None:
            self.checkpoint_name = checkpoint_name
            self.models_path = models_path
            self.load_model()
            self.create_layers()
        if model is not None:
            self.model = model
            self.create_layers()

    def load_model(self, checkpoint_name=None, models_path=None):
        if checkpoint_name is not None:
            self.checkpoint_name = checkpoint_name
        if models_path is not None:
            self.models_path = models_path
        
        if self.model_type == 'lunet':
            self.model = generate_lunet_model(
                out_channels=self.out_channels,
                checkpoint_name=self.checkpoint_name,
                models_path=self.models_path,
                device=self.device)
        elif self.model_type == 'unet':
            self.model = generate_unet_model(
                out_channels=self.out_channels,
                checkpoint_name=self.checkpoint_name,
                models_path=self.models_path,
                device=self.device)
        else:
            raise Exception(f"Model {self.model_type} not implemented yet!")
        self.model = generate_XAI_model(self.model, device=self.device).eval()

    def create_layers(self):
        try:
            layers_names = []
            for key in self.layers_dict.keys():
                layers_names.extend(self.layers_dict[key])
            self.layers = [d for d in list(
                self.model.named_modules()) if d[0] in layers_names]
        except:
            print('layers not created!')

    def run(self, image, target, method, layer_name, load_from_save=None, save_attribution=None, try_to_load=True, **kwargs):
        attr = torch.Tensor()

        if load_from_save is not None:
            try:
                #print(f'loading {load_from_save}')
                attr = torch.load(load_from_save)
                return attr
            except:
                if not try_to_load:
                    return

        if load_from_save is None or try_to_load:
            layer = [l[1] for l in self.layers if l[0] == layer_name][0]
            attr = generateAttributions(
                image, self.model, target, method, layer,device=self.device, **kwargs
            )

        if save_attribution is not None:
            try:
                save_path = "/".join(save_attribution.split('/')[:-1])
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save(attr.detach().cpu(), save_attribution)
            except:
                print(f"Save path {save_attribution} is incorrect")
        return attr
