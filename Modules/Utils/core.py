import torch 
from enum import Enum
from skimage.filters import threshold_multiotsu

def normalize_0_1(v):
    """
    normalize vector to [0,1]
    """
    return (v-v.min()) / (v.max()-v.min())


def normalize_m1_1(v):
    """
    normalize vector to [-1,1]
    """
    return 2*(v-v.min()) / (v.max()-v.min()) - 1


def normalize_a_b(v, a, b):
    """
    normalize vector to [a,b]
    formula: v_{normalized} = (b-a)\frac{v - min(v)}{max(v) - min(v)} + a
    """
    if a==b:
        print('Error: a == b')
        return    
    v_normalized = (b-a) * (v - v.min()) / (v.max() - v.min()) + a
    return v_normalized


def normalize_non_shifting_zero(v,epsilon=0,v_min = None,v_max=None):
    """
    normalize vector to [-1,1] without shifting the zero position
    """
    v_max = v.max() if v_max is None else v_max
    v_min = v.min() if v_min is None else v_min
    
    v = torch.where(abs(v)<=epsilon,0,v)
    
    if ((v_max != 0) and (v_min != 0)):
        return torch.where(v > 0, v/v_max, v/(-v_min))
    if (v_max == 0) and (v_min != 0):
        return torch.where(v > 0, v, v/(-v_min))
    if (v_max != 0) and (v_min == 0):
        return torch.where(v > 0, v/v_max, v)
    return v


def normalize_scaler(v,v_min = None,v_max=None):
    #TDOO consider adding epsilon to zero values or perfom two masks
    """
    normalize vector by deviding by the max absolute value.
    """
    v_max = v.max() if v_max is None else v_max
    v_min = v.min() if v_min is None else v_min
    Z = max(abs(v_max),abs(v_min))
    if ((v_max != 0) and (v_min != 0)):
        return v / Z
    if (v_max == 0) and (v_min != 0):
        return torch.where(v > 0, v, v/Z)
    if (v_max != 0) and (v_min == 0):
        return torch.where(v > 0, v/Z, v)
    return v

def identity(v):
    return v

def normalize_log(attr:torch.Tensor)->torch.Tensor:
    """
    Normalize the attribute tensor using log10
    """
    attr_log = torch.zeros_like(attr)
    attr_log[attr>1] = torch.log10(attr[attr>1])
    attr_log[attr<-1] = - torch.log10(-attr[attr<-1])
    return attr_log

class Normalizations(Enum):
    normalize_0_1 = 'normalize_0_1'
    normalize_m1_1 = 'normalize_m1_1'
    normalize_a_b = 'normalize_a_b'
    normalize_non_shifting_zero = 'normalize_non_shifting_zero'
    normalize_scaler = 'normalize_scaler'
    normalize_log = 'normalize_log'
    identity = 'identity'
    
    def pick(name):
        try:
            if type(name)== type(''):
                name = Normalizations[name]
            return eval(name.value)
        except:
            print('Normalization undefined')
            

def get_layer_by_name(model, layer_name):
    for name, module in model.named_modules():
        if layer_name in name:
            return module
        

def clip_fixed_percentage(v, p=0.1, replace_with_zero=False):
    new_v = v.clone().flatten()
    N = 1
    for i in range(len(new_v.shape)):
        N *= new_v.shape[i]
    x, indices = torch.sort(new_v)
    w = int(N * p)

    if len(v[v>0]) >= w:
        new_v[indices[0:w]] = x[w] if not replace_with_zero else 0
    if len(v[v<0]) >= w:
        new_v[indices[-w:]] = x[-1 - w] if not replace_with_zero else 0

    return new_v.reshape(v.shape)


def apply_otsu(attr):
    attr_pos = torch.where(attr>0,attr,0)
    attr_neg= torch.where(attr<0,attr,0)
    threshold_pos,threshold_neg = None,None
    if attr_pos.max()>0:
        try:
            threshold_pos = threshold_multiotsu(attr_pos.detach().cpu().numpy())
        except:
            threshold_pos = [attr_pos.max().item(),attr_pos.max().item()]
        if threshold_pos[1]>=attr_pos.max():
            attr_pos = torch.where(attr_pos>0,attr_pos,0)
        else:
            attr_pos = torch.where(attr_pos>threshold_pos[1],attr_pos,0)
    if attr_neg.min()<0:
        try:
            threshold_neg = threshold_multiotsu(attr_neg.detach().cpu().numpy())
        except:
            threshold_neg = [attr_neg.min().item(),attr_neg.min().item()]
        if threshold_neg[0]<=attr_neg.min():
            attr_neg = torch.where(attr_neg<0,attr_neg,0)
        else:
            attr_neg = torch.where(attr_neg<threshold_neg[0],attr_neg,0)
    attr_otsu = attr_pos + attr_neg
    return attr_otsu,{'threshold_pos':threshold_pos,'threshold_neg':threshold_neg}