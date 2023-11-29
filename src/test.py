import gc
import numpy as np
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
from torch.autograd import detect_anomaly


class CombinedHuberLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, labels, outputs):
        print(torch.exp(labels[:5]))
        print(torch.exp(outputs[:5]))
        print("###################")
        print(nn.functional.softmax(labels[:5]))
        print(nn.functional.softmax(outputs[:5]))

        loss = nn.functional.huber_loss(
            labels, outputs, delta=np.linalg.norm([1, 1, 1])
        )  # 0.1)#1.0098063938698598)

        loss_rate = nn.functional.kl_div(
            nn.functional.log_softmax(outputs), nn.functional.softmax(labels)
        )
        # loss_rate = nn.functional.huber_loss(nn.functional.softmax(outputs_rate), targets_rate, delta=0.1)
        # np.linalg.norm([2.2558166510964104, 2.1064155571070384, 1.9195553125363525]))

        # print(outputs_rate, targets_rate)
        print(loss, loss_rate)

        return loss + loss_rate


def test(
    model,
    loader: torch.utils.data.dataloader.DataLoader,
    test_gdf,
    model_path="../models/432_765/best_model",
):
    # store a string of availability of GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if device.type == "cpu":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # import the parameters of the best model
    state_dict = torch.load(model_path, map_location=torch.device(device))
    try:
        model.load_state_dict(state_dict)
    except:
        restored_state_dict = {
            k.replace("_orig_mod.", ""): v for k, v in state_dict.items()
        }
        model.load_state_dict(restored_state_dict)

    # change mode as validation
    model.eval()

    # initialise a loss function object (MSE)
    # criterion = nn.HuberLoss()
    criterion = CombinedHuberLoss()

    # set CPU or GPU
    model = model.to(device)

    # initialise a list to store the test_loss and output
    test_loss = []
    outputs = []

    with torch.no_grad():
        for i, (img, label) in enumerate(loader):
            with detect_anomaly():
                # convert images and labels to either CPU tensor or GPU tensor
                img, label = img.to(device), label.to(device)

                # execute forward propagation
                output = model(img)

                # calculate RMSE of logged labels
                loss = criterion(label, output)

                # store the test loss of this iteration
                test_loss.append(loss.item())

                # store the outputs
                outputs.append(output.cpu().detach().numpy().copy())

                # clear cache
                gc.collect()
                if device.type == "gpu":
                    torch.cuda.empty_cache()
                elif device.type == "mps":
                    torch.mps.empty_cache()
                else:
                    torch.empty_c

    # calculate mean of the test loss
    test_loss = np.mean(test_loss)

    # aggregate outputs
    outputs = np.concatenate(outputs)

    # obtain logged labels
    labels_original = test_gdf[["0_14", "15_64", "over_64"]].to_numpy()
    labels_rate = [value / np.sum(value) for value in labels_original]

    outputs_original = np.exp(outputs)
    outputs_rate = [value / np.sum(value) for value in outputs_original]

    # labels = [np.log10(value + 1e-7) for value in labels]
    # labels = np.concatenate(labels).reshape(-1, 3)
    # labels = np.maximum(labels, 0)

    # outputs_original = 10**outputs - 1e-7
    # outputs_rate = [value / np.sum(value) for value in outputs_original]

    labels = [np.log(value) for value in labels_original]
    labels = np.concatenate(labels).reshape(-1, 3)
    labels = np.maximum(labels, 0)

    labels = np.concatenate((labels, np.array(labels_rate)), axis=1)
    outputs = np.concatenate((outputs, np.array(outputs_rate)), axis=1)

    r2_scores = [r2_score(labels[:, i], outputs[:, i]) for i in range(labels.shape[1])]

    result_dict = {"outputs": outputs, "labels": labels, "r2_scores": r2_scores}

    return result_dict
