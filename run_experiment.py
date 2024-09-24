import os
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader


def train_model(model, data, labels, epochs=10, batch_size=32, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    dataset = torch.utils.data.TensorDataset(data.to(device), labels.to(device))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")

    return model


def load_model(model_path):
    model = models.resnet18(num_classes=10)  # 假设为 CIFAR-10 的 10 个类别
    model.load_state_dict(torch.load(model_path))
    return model


def train_step(step, data_dir):
    if step == 0:
        # Step 1: 训练 M_{p0}
        D_0_data = torch.load(os.path.join(data_dir, "D_0.npy"))
        D_0_labels = torch.load(os.path.join(data_dir, "D_0_labels.npy"))

        model_p0 = models.resnet18(num_classes=10)
        model_p0 = train_model(model_p0, D_0_data, D_0_labels, epochs=10)
        torch.save(model_p0.state_dict(), "model_p0.pth")
        print("M_{p0} 训练完毕并保存。")

    elif step == 1:
        # Step 2: 训练 M_{p1}
        model_p0_loaded = load_model("model_p0.pth")

        D_tr_1_data = torch.tensor(
            np.load(os.path.join(data_dir, "cifar10_D_tr_data_version_1.npy"))
        )
        D_tr_1_labels = torch.tensor(
            np.load(os.path.join(data_dir, "cifar10_D_tr_labels_version_1.npy"))
        )

        model_p1 = train_model(model_p0_loaded, D_tr_1_data, D_tr_1_labels, epochs=10)
        torch.save(model_p1.state_dict(), "model_p1.pth")
        print("M_{p1} 训练完毕并保存。")

    elif step >= 2:
        # Step 3: 迭代训练 M_{p_{2-n}}
        for i in range(2, step + 1):
            model_r_j = load_model(f"model_p{i-1}.pth")
            D_tr_i_data = torch.tensor(
                np.load(os.path.join(data_dir, f"cifar10_D_tr_data_version_{i}.npy"))
            )
            D_tr_i_labels = torch.tensor(
                np.load(os.path.join(data_dir, f"cifar10_D_tr_labels_version_{i}.npy"))
            )

            model_r_j = train_model(model_r_j, D_tr_i_data, D_tr_i_labels, epochs=10)
            torch.save(model_r_j.state_dict(), f"model_p{i}.pth")
            print(f"M_{i} 训练完毕并保存。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet models step by step.")
    parser.add_argument(
        "--step",
        type=int,
        required=True,
        help="Specify the step to execute: 0 for M_{p0}, 1 for M_{p1}, or 2+ for M_{p_{2-n}}.",
    )

    args = parser.parse_args()
    data_dir = "./data/cifar-10/noise"

    train_step(args.step, data_dir)
