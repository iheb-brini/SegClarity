from .constants import DEFAULT_DEVICE
import torch


class ModelXAI(torch.nn.Module):
  def __init__(self,pretrained_model,prediction_key=None,masking=False):
    super(ModelXAI,self).__init__()
    self.__dict__ = pretrained_model.__dict__.copy()
    self.forward_fn = pretrained_model.forward
    self.prediction_key =prediction_key 
    self.out_max = None
    self.masking = masking

  def set(self,out_max):
    self.out_max=out_max

  def reset(self):
    del self.out_max
    torch.cuda.empty_cache()
    self.out_max = None

  def check(self):
    return self.out_max is not None

  def forward(self,x):
    x= self.forward_fn(x)
    if self.prediction_key is not None:
      x = x[self.prediction_key]
    
    if not(self.check()):
      return x

    if self.out_max.shape != x.shape:
      self.out_max = torch.argmax(x, dim=1, keepdim=True)
      
      
    selected_inds = torch.zeros_like(x).scatter_(1, self.out_max, 1)

    return (x * selected_inds).sum(dim=(2,3))
    
    #h,w = x.shape[-2:]
    #return (x * selected_inds).sum(dim=(2,3)) #/ (h*w)

    #sum_selected_inds = selected_inds.sum(dim=(2,3))
    #sum_selected_inds[sum_selected_inds == 0] = torch.inf
    #return ((x * selected_inds).sum(dim=(2,3))) / sum_selected_inds
  
def generate_XAI_model(model,**kwargs):

    masking = kwargs.get('masking',False)
    prediction_key = kwargs.get('prediction_key',None)
    device = kwargs.get('device',DEFAULT_DEVICE)

    model = ModelXAI(model,prediction_key=prediction_key, masking=masking).to(device)
    return model
