import torch
import time
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import os
import pandas as pd
from tqdm import tqdm
from Modules.Architecture import generate_model
from pathlib import Path
from .tools import visualize
from PIL import Image
from Modules.Attribution import generateAttributions
from Modules.ModelXAI import generate_XAI_model
from captum.attr import LayerAttribution
from Modules.Utils import Normalizations

from skimage.filters import threshold_multiotsu

def apply_otsu(attr):
    attr_pos = torch.where(attr>0,attr,0)
    attr_neg= torch.where(attr<0,attr,0)
    threshold_pos,threshold_neg = None,None
    if attr_pos.max()>0:
        try:
            threshold_pos = threshold_multiotsu(attr_pos.detach().cpu().numpy())
        except:
            threshold_pos = [attr_pos.max().item(),attr_pos.max().item()]
        attr_pos = torch.where(attr_pos>threshold_pos[1],attr_pos,0)
    if attr_neg.min()<0:
        try:
            threshold_neg = threshold_multiotsu(attr_neg.detach().cpu().numpy())
        except:
            threshold_neg = [attr_neg.min().item(),attr_neg.min().item()]
        attr_neg = torch.where(attr_neg<threshold_neg[0],attr_neg,0)
    attr_otsu = attr_pos + attr_neg
    return attr_otsu,{'threshold_pos':threshold_pos,'threshold_neg':threshold_neg}

###########################
normalizer_no_shift = Normalizations.pick(Normalizations.normalize_non_shifting_zero)


###################
# Content heatmap #
###################


def compute_content_heatmap(attr, mask, target, normalizer=normalizer_no_shift):
    attr_normalized = normalizer(attr)
    if attr_normalized.shape != mask.shape:
        attr_normalized = LayerAttribution.interpolate(
            attr_normalized, mask.squeeze().shape[-2:]
        )
    V = attr_normalized[0][mask == target]
    if len(V) == 0:
        return 0
    return max(V.sum().item(), 0) / (len(V))


##########################
# Attribution Similarity #
##########################


def f_sum_attribution_advanced(V, sign="pos", **kwargs):
    V_pos = V[V > 0]
    V_pos_sum = V_pos.sum().item()
    V_pos_len = len(V_pos)

    V_neg = V[V < 0]
    V_neg_sum = V_neg.sum().item()
    V_neg_len = len(V_neg)

    V_zero = V[V == 0]
    V_zero_len = len(V_zero)

    if sign == "pos":
        score = V_pos_sum + V_neg_sum
    elif sign == "neg":
        score = -(V_pos_sum + V_neg_sum)
    else:
        raise Exception("Not implemented")

    return score, {
        "pos": round(V_pos_sum, 2),
        "neg": round(V_neg_sum, 2),
        "%pos": round((V_pos_len / (len(V))) * 100, 2),
        "%neg": round((V_neg_len / (len(V))) * 100, 2),
        "%zero": round((V_zero_len / (len(V))) * 100, 2),
    }


def get_component_info(attr, mask, c, sign="pos"):
    if "pos" == sign:
        V = attr[mask == c]
    else:
        V = attr[mask != c]

    if len(V) == 0:
        return 0,{
        "pos": 0,
        "neg": 0,
        "%pos": 0,
        "%neg": 0,
        "%zero": 0,
    }

    return f_sum_attribution_advanced(V, sign)


def compute_AFS(attr, mask, target):
    attr_normalized = normalizer_no_shift(attr)

    if attr_normalized.shape != mask.shape:
        attr_normalized = LayerAttribution.interpolate(
            attr_normalized, mask.squeeze().shape[-2:]
        )

    w_target = round(
        len(mask[mask == target]) / (mask.shape[-1] * mask.shape[-2]) * 100, 2
    )

    # target score
    score_target, info_target = get_component_info(
        attr_normalized[0], mask, c=target, sign="pos"
    )

    TP, FP = info_target["pos"], abs(info_target["neg"])

    # non target score
    score_non_target, info_non_target = get_component_info(
        attr_normalized[0], mask, c=target, sign="neg"
    )

    TN, FN = abs(info_non_target["neg"]), info_non_target["pos"]

    accuracy = (TP + TN) / (TP + FP + TN + FN)

    if (TP + FP) ==0:
        precision = 0
    else:
        precision = TP / (TP + FP)

    if (TP + FN) == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)

    if (precision + recall) == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    details = {
        "w_target": w_target,
        "pos_target": info_target["pos"],
        "neg_target": info_target["neg"],
        "pos_non_target": info_non_target["pos"],
        "neg_non_target": info_non_target["neg"],
        "score_target": score_target,
        "score_non_target": score_non_target,
        "acccuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    return metrics, details


def clip_fixed_percentage(v, p=0.1, replace_with_zero=False):
    new_v = v.clone().flatten()
    N = 1
    for i in range(len(new_v.shape)):
        N *= new_v.shape[i]
    x, indices = torch.sort(new_v)
    w = int(N * p)
    new_v[indices[0:w]] = x[w] if not replace_with_zero else 0
    new_v[indices[-w:]] = x[-1 - w] if not replace_with_zero else 0
    return new_v.reshape(v.shape)


target_labels = ["bg", "txt", "hl", "gl"]
###########################


class Trainer:
    def __init__(
        self,
        model,
        models_folder,
        model_type,
        model_tag,
        out_channels,
        predicions_folder,
        dataset_type,
        epochs=10,
    ) -> None:
        """_summary_

        Args:
            model (_type_): _description_
            models_folder (_type_): _description_
            model_type (_type_): _description_
            model_tag (_type_): _description_
            out_channels (_type_): _description_
            predicions_folder (_type_): _description_
            dataset_type (_type_): _description_
            epochs (int, optional): _description_. Defaults to 10.
        """
        self.model = model
        self.models_folder = models_folder
        self.model_type = model_type
        self.model_tag = model_tag
        self.out_channels = out_channels
        self.predicions_folder = predicions_folder
        self.dataset_type = dataset_type
        self.epochs = epochs

        import pandas as pd
        xai_cols = ['image_name','epoch','layer_name','method','target','afs']
        self.df_xai = pd.DataFrame(columns = xai_cols)

        # initialize optimizer and loss function
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-05
        )
        self.criterion = torch.nn.CrossEntropyLoss()

        # accumulation steps
        self.gradient_accumulation_steps = 5

        # Train the model
        self.min_loss = np.Inf
        self.best_iou_score = 0.0

        self.metrics_keys = [
            "train_loss",
            "train_accuracy",
            "train_IoU",
            *[f"train_IoU_class{i}" for i in range(out_channels)],
            "val_loss",
            "val_accuracy",
            "val_IoU",
            *[f"val_IoU_class{i}" for i in range(out_channels)],
            "time",
        ]

        self.metrics = {key: [] for key in self.metrics_keys}
        self.create_output_folders()

    def create_output_folders(self):
        for out_path in [self.models_folder, self.predicions_folder]:
            if not os.path.exists(out_path):
                os.makedirs(out_path)

    def train_one_epoch(
        self,
        epoch,
        train_loader,
        valid_loader,
        device,
        min_loss=np.Inf,
        best_iou_score=0.0,
        fit_time=time.time(),
        weighted='avg',
        xai_guided=False,
        alpha=0.1,
        highligh_only=False
    ):
        # Train the model
        trainloss = 0
        since = time.time()
        self.model.train()
        conf_mat_per_epoch = np.zeros((self.out_channels, self.out_channels))

        afs_scores = []
        afs_score = 0

        # for index, batch in enumerate(tqdm(train_loader)):
        for index, batch in enumerate((train_loader)):
            img, label, paths = batch
            """
            Traning the Model.
            """
            image_name = paths[0][0].split('/')[-1].split('.')[0]

            img = img.float()
            img = img.to(device)
            label = label.to(device)
            output = self.model(img)
            loss = self.criterion(output, label)
            label_size = (label.shape[-1] * label.shape[-2])
            w_target = {}
            for target  in [0,1,2,3]:
                w_target[target] = len(label[label == target]) / label_size


            #xai_guided = epoch > 9
            if xai_guided == True:

                d = {}
                method = "LayerGradCam"
                #layer_names = ['conv1.0','conv2.0','conv3.0','conv4.0',"final_layer"]
                layer_names = ["final_layer"]
                sum_score = 0.0 # accross different classes
                targets  = [2] if highligh_only else [i for i in range(self.out_channels)]

                scores_dict = {layer_name:{} for layer_name in layer_names}
                for layer_name in layer_names:
                    for target in targets:
                        xai_model = generate_XAI_model(self.model, device=device).eval()
                        _, layer = [
                            d for d in list(xai_model.named_modules()) if d[0] == layer_name
                        ][0]
                        attr = generateAttributions(
                            img,
                            xai_model,
                            target,
                            method,
                            layer,
                            device=device,
                        )

                        attr,_= apply_otsu(attr)

                        eps = 0.01
                        if abs(attr.min()) < eps and abs(attr.max()) < eps:
                            score = 0
                        else:
                            metrics, infos = compute_AFS(attr, label, target)
                            score = metrics["f1"]
                        d[target] = score
                        
                        # save the AFS score for layer_name and target
                        scores_dict[layer_name][target] = score

                        sum_score+= score

                        self.df_xai.loc[len(self.df_xai)] = {'image_name':image_name,
                                                            'epoch':epoch,
                                                            'layer_name':layer_name,
                                                            'target':target,
                                                            'afs':score,
                                                            }
                with_weighted_score = (weighted == 'weighted')
                if with_weighted_score:
                    sum_of_inverse_sum = 0.0
                    weighted_avg_afs_score = 0.0
                    for target in targets:
                        if w_target[target] !=0:
                            weighted_avg_afs_score+= scores_dict[layer_names[0]][target] / w_target[target]
                            sum_of_inverse_sum+= 1 / w_target[target]
                    xai_loss = 1 - weighted_avg_afs_score / sum_of_inverse_sum
                else:
                    sum_score = 0.0
                    n_available_classes = 0
                    for target in targets:
                        if w_target[target] !=0:
                            sum_score+=scores_dict[layer_names[0]][target]
                            n_available_classes +=1
                    try:
                        xai_loss = 1 - sum_score / n_available_classes
                    except:
                        xai_loss = 1 - sum_score

                #for name, param in self.model.named_parameters():
                #    for layer_name in layer_names:
                #        if layer_name in name:
                #            # updated gradient                    
                #            param.grad += alpha * (1 - sum_score / len(targets))

                #afs_score = sum(list(d.values())) / len(list(d.values()))
                #afs_scores.append(d.copy())
                # print(f"Score {score}")

                loss = loss + alpha * xai_loss
            loss.backward()


            if (index + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            trainloss += loss.item()
            output = F.softmax(output, dim=1)
            output = torch.argmax(output, dim=1)
            output = output.cpu().flatten()
            label = label.cpu().flatten()
            conf_mat = confusion_matrix(
                label, output, labels=[i for i in range(self.out_channels)]
            )

            conf_mat_per_epoch += np.array(conf_mat)


        A_inter_B = conf_mat_per_epoch.diagonal()
        A = conf_mat_per_epoch.sum(1)
        B = conf_mat_per_epoch.sum(0)
        Jaccard_classes = A_inter_B / (A + B - A_inter_B)
        macro_jaccard = np.nanmean(Jaccard_classes)
        Accuracy_classes = conf_mat_per_epoch.diagonal() / conf_mat_per_epoch.sum(1)
        macro_accuracy = np.nanmean(Accuracy_classes)

        # set the model in evaluation mode
        self.model.eval()
        valloss = 0
        # switch off autograd
        with torch.no_grad():
            # loop over the validation set
            conf_mat_per_epoch_val = np.zeros((self.out_channels, self.out_channels))

            # for img_val,label_val,_ in tqdm(valid_loader):
            for img_val, label_val, _ in valid_loader:
                """
                Validation of Model.
                """
                # send the input to the device
                img_val = img_val.float()
                img_val = img_val.to(device)
                label_val = label_val.to(device)
                # make the prediction and calculate the validation loss
                output_val = self.model(img_val)
                loss_val = self.criterion(output_val, label_val)
                # add the loss to the total validation loss so far
                valloss += loss_val.item()
                output_val = F.softmax(output_val, dim=1)
                output_val = torch.argmax(output_val, dim=1)
                output_val = output_val.cpu().flatten()
                label_val = label_val.cpu().flatten()
                conf_mat_val = confusion_matrix(
                    label_val, output_val, labels=[i for i in range(self.out_channels)]
                )
                conf_mat_per_epoch_val += np.array(conf_mat_val)

        A_inter_B_val = conf_mat_per_epoch_val.diagonal()
        A_val = conf_mat_per_epoch_val.sum(1)
        B_val = conf_mat_per_epoch_val.sum(0)
        Jaccard_classes_val = A_inter_B_val / (A_val + B_val - A_inter_B_val)
        macro_jaccard_val = np.nanmean(Jaccard_classes_val)
        Accuracy_classes_val = (
            conf_mat_per_epoch_val.diagonal() / conf_mat_per_epoch_val.sum(1)
        )
        macro_accuracy_val = np.nanmean(Accuracy_classes_val)

        # append the average training and validation information
        self.metrics["train_loss"].append(trainloss / len(train_loader))
        self.metrics["train_accuracy"].append(macro_accuracy)
        self.metrics["train_IoU"].append(macro_jaccard)

        self.metrics["val_loss"].append(valloss / len(valid_loader))
        self.metrics["val_accuracy"].append(macro_accuracy_val)
        self.metrics["val_IoU"].append(macro_jaccard_val)

        for i in range(self.out_channels):
            self.metrics[f"train_IoU_class{i}"].append(Jaccard_classes[i])
            self.metrics[f"val_IoU_class{i}"].append(Jaccard_classes_val[i])

        print("Total time: {:.2f} m".format((time.time() - fit_time) / 60))
        self.metrics["time"].append((time.time() - since) / 60)

        subpaths_to_saves = [
            "per_epoch_models",
            "ToTest",
            "saved_min_loss_models",
            "saved_best_iou_models",
        ]
        for subpath in subpaths_to_saves:
            path_sub = f"{self.models_folder}/{subpath}"
            if not os.path.exists(path_sub):
                os.makedirs(path_sub)

        # Save model per epoch
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
            },
            f"{self.models_folder}/per_epoch_models/model" + str(epoch + 1) + ".pt",
        )
        print("Model for epoch" + str(epoch + 1) + " saved!")

        # Save model per epoch for test
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
            },
            f"{self.models_folder}/ToTest/model" + str(epoch + 1) + ".pt",
        )

        # Save model if a min val loss score is obtained
        if min_loss > (valloss / len(valid_loader)):
            min_loss = valloss / len(valid_loader)
            torch.save(
                {
                    "epoch": self.epochs,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": self.criterion,
                },
                f"{self.models_folder}/saved_min_loss_models/loss_unet_{self.epochs}epochs.pt",
            )
            print("min Loss_Model saved!")

        # Save model if a better val IoU score is obtained
        if best_iou_score < (macro_jaccard_val):
            best_iou_score = macro_jaccard_val
            torch.save(
                {
                    "epoch": self.epochs,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": self.criterion,
                },
                f"{self.models_folder}/saved_best_iou_models/IoU_unet_{self.epochs}epochs.pt",
            )
            print("best IOU_Model saved!")

        return min_loss, best_iou_score, fit_time

    def test_one_epoch(self, epoch, test_loader, device_test, label_colors=None):
        """_summary_

        Args:
            test_loader (_type_): _description_
            device_test (_type_): _description_
        """
        Test_model = generate_model(
            model_type=self.model_type,
            out_channels=self.out_channels,
            device=device_test,
        )

        Test_checkpoint_iou = torch.load(
            f"{self.models_folder}/ToTest/model" + str(epoch + 1) + ".pt",
            map_location="cpu",
        )

        # load model weights state_dict
        Test_model.load_state_dict(Test_checkpoint_iou["model_state_dict"])
        # print("Previously trained model weights state_dict loaded...")

        # create folder to put the predictions in
        output_path = Path(f"{self.predicions_folder}/model{str(epoch + 1)}")

        for index, batch in enumerate((test_loader)):
            img, label, paths = batch
            chunks = paths[0][0].split("/")[-1].split(".")
            image_name, ext = chunks[0], chunks[-1]

            inputs = img.to(device_test)
            outputs = Test_model(inputs.float())
            pr_mask_show = outputs.squeeze().cpu().detach().numpy()
            pr_mask_show_2 = np.argmax(pr_mask_show.transpose((1, 2, 0)), axis=2)
            diva_out = (
                visualize(pr_mask_show_2, self.dataset_type, label_colors=label_colors)
                * 255
            ).astype(np.uint8)
            diva_out = Image.fromarray(diva_out)
            output_path.mkdir(exist_ok=True)
            file_path = os.path.join(output_path, image_name)
            diva_out.save((file_path + ".gif"))

    def train(self, train_loader, valid_loader, device, xai_guided=False):
        min_loss = np.Inf
        best_iou_score = 0.0
        fit_time = time.time()

        # loop over epochs
        for epoch in tqdm(range(self.epochs)):
            (min_loss, best_iou_score, fit_time) = self.train_one_epoch(
                epoch,
                train_loader,
                valid_loader,
                device,
                min_loss,
                best_iou_score,
                fit_time,
                xai_guided=xai_guided,
            )

    def test(self, test_loader, device_test, label_colors=None):
        # loop over epochs
        for epoch in tqdm(range(self.epochs)):
            self.test_one_epoch(
                epoch, test_loader, device_test, label_colors=label_colors
            )

    def train_and_test(
        self,
        train_loader,
        valid_loader,
        test_loader,
        device_train,
        device_test,
        label_colors=None,
        weighted='avg',
        xai_guided=False,
        alpha=0.1,
        starting_epoch=0,
        highligh_only=False
    ):
        min_loss = np.Inf
        best_iou_score = 0.0
        fit_time = time.time()
        print("Starting training....")
        for epoch in tqdm(range(self.epochs)):
            (min_loss, best_iou_score, fit_time) = self.train_one_epoch(
                epoch,
                train_loader,
                valid_loader,
                device_train,
                min_loss,
                best_iou_score,
                fit_time,
                weighted=weighted,
                xai_guided=xai_guided and (epoch >=starting_epoch),
                alpha=alpha,
                highligh_only=highligh_only
            )

            self.test_one_epoch(
                epoch, test_loader, device_test, label_colors=label_colors
            )
            self.generate_csv()

        print("Training finished!")

        self.generate_csv()
        print("Metrics generated!")

    def generate_csv(self,**kwargs):
        # save metrics file
        data = pd.DataFrame({**{key: self.metrics[key] for key in self.metrics_keys}})

        from datetime import datetime

        pad_with_zero = lambda s: "0" * (2 - len(str(s))) + str(s)
        current_date = datetime.now()
        day = pad_with_zero(current_date.day)
        month = pad_with_zero(current_date.month)
        year = pad_with_zero(current_date.year)

        data.to_excel(
            f"{self.models_folder}/loss+metrics_{self.model_type}_{self.model_tag}_{self.epochs}epochs_{day}-{month}-{year}.xlsx",
            index=False,
        )

        if len(self.df_xai)>0:
            self.df_xai.to_excel(
            f"{self.models_folder}/loss+xai_{self.model_type}_{self.model_tag}_{self.epochs}epochs_{day}-{month}-{year}.xlsx",
            index=False,
            )