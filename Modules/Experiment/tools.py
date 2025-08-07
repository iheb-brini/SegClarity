from Modules.Evaluation import (
    ComplexityMetric,
    CoherencyMetric,
    AverageDropMetric,
    AverageDropIoUMetric,
    ADCCMetric,
    calculate_explanation_map,
)
from Modules.Utils import Normalizations
from captum.attr import LayerAttribution

DEFAULT_NORMALIZER = Normalizations.pick(Normalizations.normalize_0_1)


def generate_config(
    models_dir,
    model_tag,
    checkpoint_tag,
    attribution_dir,
    image_name,
    layer_name,
    target,
    method,
    **kwargs,
):
    checkpoint_name = f"{checkpoint_tag}.pt"
    model_path = f"{models_dir}/{model_tag}"
    file_directory = (
        f"{attribution_dir}/{model_tag}/{checkpoint_tag}/{image_name}/{method}"
    )
    n_steps = kwargs.get("n_steps", None)
    if n_steps is None:
        file_path = f"{file_directory}/{layer_name}_{target}.pt"
    else:
        file_path = f"{file_directory}/{layer_name}_ns_{n_steps}_{target}.pt"

    return checkpoint_name, model_path, file_path


def generateParam_adcc(
    inp,
    experiment,
    target,
    method,
    layer_name,
    device,
    normalizer=DEFAULT_NORMALIZER,
    nc=4,
    **kwargs,
):

    attribution_map_path = kwargs.get("attribution_map_path", None)
    explanation_map_path = kwargs.get("explanation_map_path", None)

    params = {}
    params["target"] = target
    params["num_classes"] = nc

    params["pred"] = experiment.model.to(device)(inp) #.cpu()

    params["attr"] = experiment.run(
        inp,
        target=target,
        method=method,
        layer_name=layer_name,
        #load_from_save=attribution_map_path,
        #try_to_load=False,
    ) #.cpu()
    params["attr"] = LayerAttribution.interpolate(
        params["attr"], inp.squeeze().shape[1:]
    )

    explanation_map = calculate_explanation_map(inp, params["attr"], normalizer, device) #.cpu()

    params["pred_on_exp"] = experiment.model(explanation_map)
    #params["pred_on_exp"] = experiment.model.cpu()(explanation_map).cpu()

    params["attr_on_exp"] = experiment.run(
        explanation_map,
        target=target,
        method=method,
        layer_name=layer_name,
        #load_from_save=explanation_map_path,
        #try_to_load=False,
    ) #.cpu()

    params["attr_on_exp"] = LayerAttribution.interpolate(
        params["attr_on_exp"], inp.squeeze().shape[1:]
    ) #.cpu()

    return params


def generateParam_adcc_iou(
    inp,
    experiment,
    target,
    method,
    layer_name,
    device,
    normalizer=DEFAULT_NORMALIZER,
    nc=4,
):
    params = {}
    params["target"] = target
    params["num_classes"] = nc

    params["out"] = experiment.model(inp).max(-3)[1]
    params["attr"] = experiment.run(
        inp, target=target, method=method, layer_name=layer_name
    )

    params["attr"] = LayerAttribution.interpolate(
        params["attr"], inp.squeeze().shape[1:]
    )

    explanation_map = calculate_explanation_map(inp, params["attr"], normalizer, device)

    params["out_on_exp"] = experiment.model(explanation_map).max(-3)[1]

    params["attr_on_exp"] = experiment.run(
        explanation_map, target=target, method=method, layer_name=layer_name
    )
    params["attr_on_exp"] = LayerAttribution.interpolate(
        params["attr_on_exp"], inp.squeeze().shape[1:]
    )
    return params


def evalCam(params, normalize_map):
    com = ComplexityMetric(normalize_map).calculate(params["attr"])

    avgdrop = AverageDropMetric().calculate(
        params["pred"], params["pred_on_exp"], params["target"]
    )
    coh = CoherencyMetric().calculate(params["attr"], params["attr_on_exp"])

    adcc = ADCCMetric().calculate(com, avgdrop, coh)

    return {"com": com, "avgdrop": avgdrop, "coh": coh, "adcc": adcc}


def evalCam_iou(params, normalize_map):
    com = ComplexityMetric(normalize_map).calculate(params["attr"])

    avgdrop_iou = AverageDropIoUMetric(num_classes=params["num_classes"]).calculate(
        params["pred"], params["pred_on_exp"], params["target"]
    )
    coh = CoherencyMetric().calculate(params["attr"], params["attr_on_exp"])

    adcc = ADCCMetric().calculate(com, avgdrop_iou, coh)

    return {"com": com, "avgdrop_iou": avgdrop_iou, "coh": coh, "adcc_iou": adcc}


def evalCam_mixed(params, normalize_map):
    com = ComplexityMetric(normalize_map).calculate(params["attr"])

    avgdrop = AverageDropMetric().calculate(
        params["pred"], params["pred_on_exp"], params["target"]
    )

    avgdrop_iou = AverageDropIoUMetric(num_classes=params["num_classes"]).calculate(
        params["pred"].max(-3)[1], params["pred_on_exp"].max(-3)[1], params["target"]
    )

    coh = CoherencyMetric().calculate(params["attr"], params["attr_on_exp"])

    adcc = ADCCMetric().calculate(com, avgdrop, coh)
    adcc_iou = ADCCMetric().calculate(com, avgdrop_iou, coh)

    return {
        "com": com,
        "avgdrop": avgdrop,
        "avgdrop_iou": avgdrop_iou,
        "coh": coh,
        "adcc": adcc,
        "adcc_iou": adcc_iou,
    }
