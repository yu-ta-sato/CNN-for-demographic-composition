import os
import gc
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# class CombinedHuberLoss(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, targets, outputs):
#         targets_rate = torch.pow(10, targets) / torch.pow(10, targets).sum(
#             axis=-1
#         ).unsqueeze(1).repeat(1, 3)
#         outputs_rate = torch.pow(10, outputs) / torch.pow(10, outputs).sum(
#             axis=-1
#         ).unsqueeze(1).repeat(1, 3)

#         print(targets_rate)
#         print(outputs_rate)

#         loss = torch.nn.functional.huber_loss(
#             targets,
#             outputs,
#             delta=np.linalg.norm(
#                 [2.2558166510964104, 2.1064155571070384, 1.9195553125363525]
#             ),
#         )
#         loss_rate = torch.nn.functional.huber_loss(
#             targets_rate,
#             outputs_rate,
#             delta=np.linalg.norm([0.038376, 0.089343, 0.113496]),
#         )

#         # print(outputs_rate, targets_rate)
#         print(loss, loss_rate)

#         return loss + loss_rate


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


def train(
    model,
    train_loader: torch.utils.data.dataloader.DataLoader,
    valid_loader: torch.utils.data.dataloader.DataLoader,
    epochs: int,
    early_stopping: int,
    record_path="../src/pretrained/record_dict_pkl",
    latest_path="../src/pretrained/latest_model",
    best_path="../src/pretrained/best_model",
):
    """
    A function of managing the training of CNN model at once.

    Args:
       model           : An object of RemoteSensingCNN.
       train_loader    : A torch.utils.data.dataloader.DataLoader object for training dataset.
       valid_loader    : A torch.utils.data.dataloader.DataLoader object for validation dataset.
       epochs          : An integer which specifies the number of total epoches.
       early_stopping  : An integer which spcefifies the torrelance of early stopping.
       record_path     : By default, it is '../src/pretrained/record_dict_pkl'.
       latest_path     : By default, it is '../src/pretrained/latest_model'.
       best_path       : By default, it is '../src/pretrained/best_model'.

    Returns:
        None
    """

    # store a string of availability of GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if device.type == "cpu":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # if there is already a recorded file, import it together with the latest model
    if os.path.exists(record_path):
        with open(record_path, "rb") as f:
            record_dict = pickle.load(f)
        best_loss = min([v["valid_loss"] for v in record_dict.values()])
        not_improving = record_dict[max(record_dict.keys())]["not_improving"]
        model.load_state_dict(
            torch.load(latest_path, map_location=torch.device(device))
        )

    # initiate the record dictornary and best loss as positive infinite
    else:
        record_dict = {}
        best_loss = np.inf
        not_improving = 0

    # initialise a loss function object (MSE)
    # criterion = nn.MSELoss()

    # initialise a loss functio object (Huber loss)
    # criterion = nn.HuberLoss()
    criterion = CombinedHuberLoss()

    # initialise an optimiser object (Adam)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # set CPU or GPU
    model = model.to(device)

    # complie the model
    # model = torch.compile(model)

    # iterate over epochs
    for epoch in range(epochs):
        # set seed by epoch
        torch.manual_seed(epoch)

        # skip completed epoches
        if epoch in record_dict.keys():
            continue

        # change mode as training
        model.train(True)

        # initialise a list for training loss
        train_loss = []

        for i, (img, label) in enumerate(train_loader):
            # with detect_anomaly():
            # convert images and labels to either CPU tensor or GPU tensor
            # img, label = img.to(device), label.to(device)
            img, label = (img.to(device), label.to(device))

            # clean up the gradient
            optimizer.zero_grad()

            # execute forward propagation
            output = model(img)

            # if the output is NaN, skip back propagation
            if torch.isnan(output).any():
                print(f"[epoch {epoch + 1},{i + 1:3d}] loss: skipped due to nan")

            else:
                # calculate RMSE of logged labels
                # loss = torch.sqrt(criterion(label, output))

                # calculate log cosh
                loss = criterion(label, output)

                # execute back propagation
                loss.backward()
                optimizer.step()

                # store the training loss of this iteration
                train_loss.append(loss.item())

                print(f"[epoch {epoch + 1},{i + 1:3d}] loss: {loss.item():.3f}")

        # calculate mean of the training loss
        train_loss = np.mean(train_loss)

        # release memory on GPU
        if device == "cuda:0" or device.type == "mps":
            gc.collect()
            torch.cuda.empty_cache()

        # change mode as validation
        model.eval()

        # initialise a list for validation loss
        valid_loss = []
        with torch.no_grad():
            for i, (img, label) in enumerate(valid_loader):
                # with detect_anomaly():
                # convert images and labels to either CPU tensor or GPU tensor
                # img, label = img.to(device), label.to(device)
                img, label = (img.to(device), label.to(device))

                # execute forward propagation
                output = model(img)

                # calculate Huber loss & Cross Entropy loss
                loss = criterion(label, output)

                # store the validation loss of this iteration
                valid_loss.append(loss.item())

        # calculate mean of the validation loss
        valid_loss = np.mean(valid_loss)

        # if the validation loss is the best so far, store it as best_loss and save the model
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), best_path)
            not_improving = 0
        else:
            not_improving += 1

        # update the record dictionary
        record_dict.update(
            {
                epoch: {
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "not_improving": not_improving,
                }
            }
        )

        print(
            f"[epoch {epoch + 1}] train_loss: {train_loss:.3f}, valid_loss: {valid_loss:.3f}, not_improving: {not_improving}"
        )

        # save record_dict as pickle
        with open(record_path, "wb") as f:
            pickle.dump(record_dict, f)

        # save the latest model
        torch.save(model.state_dict(), latest_path)

        if not_improving == early_stopping:
            print("Early Stopping...")
            break


if __name__ == "__main__":
    master_gdf = gpd.read_file("../data/census/master_gdf.gpkg")
    remote_sensing_cnn = RemoteSensingCNN(out_dim=3, pretrained=True)

    key_codes = (
        master_gdf[["0_14", "15_64", "over_64", "KEY_CODE"]]
        .groupby("KEY_CODE")
        .mean()
        .sum(axis=1)
    )
    key_deciles = pd.qcut(key_codes, 10, labels=False)

    train_key_codes, valid_key_codes = train_test_split(
        key_deciles, test_size=0.2, random_state=1, stratify=key_deciles
    )
    valid_key_codes, test_key_codes = train_test_split(
        valid_key_codes, test_size=0.5, random_state=1, stratify=valid_key_codes
    )

    train_gdf = master_gdf[master_gdf["KEY_CODE"].isin(train_key_codes.index)]
    valid_gdf = master_gdf[master_gdf["KEY_CODE"].isin(valid_key_codes.index)]
    test_gdf = master_gdf[master_gdf["KEY_CODE"].isin(test_key_codes.index)]

    train_dataset = RemoteSensingDataset(train_gdf, size=334)
    valid_dataset = RemoteSensingDataset(valid_gdf, size=334)
    test_dataset = RemoteSensingDataset(test_gdf, size=334)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    train(remote_sensing_cnn, train_loader, valid_loader, epochs=3, early_stopping=30)
