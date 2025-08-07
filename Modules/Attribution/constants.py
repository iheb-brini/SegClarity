from torch.cuda import is_available

DEFAULT_DEVICE = "cuda:0" if is_available() else "cpu"

METHOD_NAMES = ['LayerLRP', 'LayerGradCam',
                'LayerDeepLift', 'LayerGradientXActivation']

lunet_encoder_layers = [*[f'dil{i}.10' for i in range(1, 5)]]
lunet_encoder_relu_layers = [*[f'dil{i}.11' for i in range(1, 5)]]
lunet_decoder_layers = [*[f'conv{i}.1' for i in range(1, 4)]] + ['conv4.1','conv4.4']
lunet_decoder_relu_layers = [
    *[f'conv{i}.2' for i in range(1, 4)]] + ['conv4.5']
lunet_special_layers = [*['bott.4'], 'final_layer',]

LUNET_LAYERS = {
    'encoder_layers': lunet_encoder_layers,
    'decoder_layers': lunet_decoder_layers,
    'special_layers': lunet_special_layers,
}
activation_layers = []
for i in range(1, 5):
    activation_layers += [*[f'dil{i}.{k}' for k in range(11)]]


activation_layers += [*[f'bott.{i}' for i in range(6)]]

for i in range(1, 4):
    activation_layers += [*[f'conv{i}.{k}' for k in range(3)]]

activation_layers += [*[f'conv4.{k}' for k in range(6)]]
activation_layers += [*['final_layer']]

LUNET_ACTIVATION_LAYERS = {
    'activation_layers': activation_layers
}
lunet_ordered_layers = []
L = len(lunet_encoder_layers)
for i in range(L):
    lunet_ordered_layers.append(lunet_encoder_layers[L - i - 1])
    lunet_ordered_layers.append(lunet_decoder_layers[i])

lunet_ordered_layers = [LUNET_LAYERS['special_layers'][0]] +\
    lunet_ordered_layers +\
    [LUNET_LAYERS['special_layers'][-1]]

labels = [
    'bottleneck',
    'encoder_4',
    'decoder_4',
    'encoder_3',
    'decoder_3',
    'encoder_2',
    'decoder_2',
    'encoder_1',
    'decoder_1.1',
    'decoder_1.2',
    'final_layer'
]

labels_act = [
    'activation'
]

##############
# IDCAR 2024 #
##############
LUNET_LAYERS_JOURNAL = {
    # dec4
    "conv1.0": "DEC4 conv",
    # dec3
    "conv2.0": "DEC3 conv",
    # dec2
    "conv3.0": "DEC2 conv",
    # dec1
    "conv4.0": "DEC1 conv",
    # final layer
    "final_layer": "final_layer",
}



LUNET_LAYERS_ICDAR = {
    # dec4
    "conv1.0": "DEC4 conv",
    "conv1.1": "DEC4 BN",
    "conv1.2": "DEC4 relu",
    # dec3
    "conv2.0": "DEC3 conv",
    "conv2.1": "DEC3 BN",
    "conv2.2": "DEC3 relu",
    # dec2
    "conv3.0": "DEC2 conv",
    "conv3.1": "DEC2 BN",
    "conv3.2": "DEC2 relu",
    # dec1
    "conv4.0": "DEC1 conv",
    "conv4.1": "DEC1 BN",
    "conv4.2": "DEC1 relu",
    # final layer
    "final_layer": "final_layer",
}




LUNET_LAYERS_EXTENDED_ICDAR = {
    #ENC1
    "dil1.0":"ENC1 conv1",
    "dil1.9":"ENC1 conv4",
    "dil1.10":"ENC1 BN4",
    "dil1":"ENC1",

    #ENC2
    "dil2.0":"ENC2 conv1",
    "dil2.9":"ENC2 conv4",
    "dil2.10":"ENC2 BN4",
    "dil2":"ENC2",

    #ENC3
    "dil3.0":"ENC3 conv1",
    "dil3.9":"ENC3 conv4",
    "dil3.10":"ENC3 BN4",
    "dil3":"ENC3",


    #ENC4
    "dil4.0":"ENC4 conv1",
    "dil4.9":"ENC4 conv4",
    "dil4.10":"ENC4 BN4",
    "dil4":"ENC4",

    #bott
    "bott.0":"BOTT conv",
    "bott.3":"BOTT conv2",
    "bott.4":"BOTT BN2",
    "bott":"BOTT",

    #DEC1
    "conv4.0": "DEC1 conv1",
    "conv4.1": "DEC1 BN4",
    "conv4.2": "DEC1 relu",
    "conv4.3": "DEC1 conv2",
    "conv4.4": "DEC1 BN2",
    "conv4":"DEC1",

    #DEC2
    "conv3.0": "DEC2 conv",
    "conv3.1": "DEC2 BN",
    "conv3":"DEC2",
 
    #DEC3
    "conv2.0": "DEC3 conv",
    "conv2.1": "DEC3 BN",
    "conv2":"DEC3",

    #DEC4
    "conv1.0": "DEC4 conv",
    "conv1.1": "DEC4 BN",
    "conv1":"DEC4",


    #DECONV
    "tconv1":"tconv1",
    "tconv2":"tconv2",
    "tconv3":"tconv3",
    "tconv4":"tconv4",

    #final layer
    "final_layer": "final_layer",
}


UNET_LAYERS_JOURNAL = {
    # dec4
    "conv5.0": "DEC4 conv",
    # dec3
    "conv6.0": "DEC3 conv",
    # dec2
    "conv7.0": "DEC2 conv",
    # dec1
    "conv8.0": "DEC1 conv",
    # final layer
    "final_layer": "final_layer",
}



UNET_LAYERS_ICDAR = {
    # dec4
    "conv5.0": "DEC4 conv",
    "conv5.1": "DEC4 BN",
    "conv5.2": "DEC4 relu",
    # dec3
    "conv6.0": "DEC3 conv",
    "conv6.1": "DEC3 BN",
    "conv6.2": "DEC3 relu",
    # dec2
    "conv7.0": "DEC2 conv",
    "conv7.1": "DEC2 BN",
    "conv7.2": "DEC2 relu",
    # dec1
    "conv8.0": "DEC1 conv",
    "conv8.1": "DEC1 BN",
    "conv8.2": "DEC1 relu",
    # final layer
    "final_layer": "final_layer",
}



UNET_LAYERS_EXTENDED_ICDAR = {
    #ENC1
    "conv1.0":"ENC1 conv1",
    "conv1.3":"ENC1 conv2",
    "conv1.4":"ENC1 BN2",
    "conv1":"ENC1",

    #ENC2
    "conv2.0":"ENC2 conv1",
    "conv2.3":"ENC2 conv2",
    "conv2.4":"ENC2 BN2",
    "conv2":"ENC2",

    #ENC3
    "conv3.0":"ENC3 conv1",
    "conv3.3":"ENC3 conv2",
    "conv3.4":"ENC3 BN2",
    "conv3":"ENC3",

    #ENC4
    "conv4.0":"ENC4 conv1",
    "conv4.3":"ENC4 conv2",
    "conv4.4":"ENC4 BN2",
    "conv4":"ENC4",


    #bott
    "bottleneck.0":"BOTT conv",
    "bottleneck.3":"BOTT conv2",
    "bottleneck.4":"BOTT BN2",
    "bottleneck":"BOTT",

    #DEC1
    "conv8.0": "DEC1 conv1",
    "conv8.3": "DEC1 conv2",
    "conv8.4": "DEC1 BN2",
    "conv8":"DEC1",

    #DEC2
    "conv7.0": "DEC2 conv",
    "conv7.3": "DEC1 conv2",
    "conv7.4": "DEC2 BN",
    "conv7":"DEC2",
 
    #DEC3
    "conv6.0": "DEC3 conv",
    "conv6.3": "DEC1 conv2",
    "conv6.4": "DEC3 BN",
    "conv6":"DEC3",

    #DEC4
    "conv5.0": "DEC4 conv",
    "conv5.3": "DEC1 conv2",
    "conv5.4": "DEC4 BN",
    "conv5":"DEC4",

    #DECONV
    "tconv1":"tconv1",
    "tconv2":"tconv2",
    "tconv3":"tconv3",
    "tconv4":"tconv4",

    #final layer
    "final_layer": "final_layer",
}
