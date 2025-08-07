import torch
from captum.attr._utils.lrp_rules import EpsilonRule,IdentityRule


def access(idx, x):
    if idx == 0:
        return str(x)
    elif not x.isnumeric() and len(str(x)) > 1:
        return str(f".{x}")
    else:
        return f"[{x}]"

def applyLRPrules(model: torch.nn.Module):
    for name, module in model.named_modules():
        if type(module) == torch.nn.modules.conv.ConvTranspose2d:
            setattr(eval(f"model.{name}"), "rule", EpsilonRule())

        elif type(module) == torch.nn.modules.activation.ReLU:
            chunks = name.split('.')
            access_name = "".join([access(idx, x)
                                   for idx, x in enumerate(chunks)])
            setattr(eval(f"model.{access_name}"), "rule", IdentityRule())

        elif type(module) == torch.nn.modules.Identity:
            chunks = name.split('.')
            access_name = "".join([access(idx, x)
                                   for idx, x in enumerate(chunks)])
            setattr(eval(f"model.{access_name}"), "rule", IdentityRule())

        elif type(module) == torch.nn.modules.instancenorm.InstanceNorm2d:
            chunks = name.split('.')
            access_name = "".join([access(idx, x)
                                   for idx, x in enumerate(chunks)])
            setattr(eval(f"model.{access_name}"), "rule", EpsilonRule())
    return model