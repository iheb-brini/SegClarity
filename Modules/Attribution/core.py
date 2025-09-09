import torch

from captum.attr import (LayerGradCam, LayerDeepLift, LayerGradientXActivation, LayerLRP,
                         LayerGradientShap, LayerDeepLiftShap,
                         LayerActivation,LayerConductance
                         )
from .tools import applyLRPrules
from .constants import DEFAULT_DEVICE


def generateAttributions(x, model, target, method, layer,device=DEFAULT_DEVICE, debug=False, **kwargs):    
    model.reset()
    if type(method) == type(""):
        method = eval(method)
    if type(layer) == type(""):
        layer = eval(layer)

    model_out = model(x)
    out_max = torch.argmax(model_out, dim=1, keepdim=True)
    del model_out

    if method == LayerLRP:
        model = applyLRPrules(model=model)

    model.set(out_max)
    lgc = method(model, layer)
    if method in [LayerGradientShap, LayerDeepLiftShap]:
        rand_img_dist = torch.cat([x * 0, x * 1])
        gc_attr = lgc.attribute(x, rand_img_dist, target=target)

    elif method == LayerLRP:
        gc_attr = lgc.attribute(x, target=target)
        
    elif method == LayerConductance:
        n_steps = kwargs.get('n_steps',5)
        gc_attr = lgc.attribute(x, target=target,n_steps=n_steps)

    else:

        if method == LayerDeepLift:
            baselines = kwargs.get(
                'baselines', torch.ones(x.shape) * 0)
            # baselines = torch.ones(x.shape) * 50 / 255
            gc_attr = lgc.attribute(
                x, target=target, baselines=baselines.to(device))
        elif method == LayerGradCam:
            relu_attributions = kwargs.get('relu_attributions',False)
            attr_dim_summation = kwargs.get('attr_dim_summation',True)
            gc_attr = lgc.attribute(x, target=target,relu_attributions=relu_attributions,attr_dim_summation=attr_dim_summation)
        else:
            gc_attr = lgc.attribute(x, target=target)


    model.reset()

    del out_max
    #torch.cuda.empty_cache()
    
    if debug:
        la = LayerActivation(model, layer)
        activation = la.attribute(x)
        print("Input Shape:", x.shape)
        print("Layer Activation Shape:", activation.shape)
        print("Layer Attribution Shape:", gc_attr.shape)

        print("Mean", gc_attr.mean())
        print("Max", gc_attr.max())
        print("Min", gc_attr.min())

    return gc_attr.sum(dim=1, keepdim=True)
