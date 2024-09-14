import torch
import os
from tqdm import tqdm
import json


class TrainTestUtils:
    def __init__(self, model_name, dataset_name):
        self.model_name = model_name
        self.dataset_name = dataset_name

    def create_save_path(self, condition):
        save_dir = os.path.join("models", self.model_name, self.dataset_name, condition)
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def l1_regularization(self, model):
        params_vec = []
        for param in model.parameters():
            params_vec.append(param.view(-1))
        return torch.linalg.norm(torch.cat(params_vec), ord=1)

    def train_and_save(
        self,
        model,
        train_loader,
        criterion,
        optimizer,
        save_path,
        epoch,
        num_epochs,
        save_final_model_only=True,
        **kwargs,  # 捕获额外的训练参数
    ):
        """
        :param save_final_model_only: If True, only save the model after the final epoch.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)  # 确保模型移动到正确的设备
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        # 提取 kwargs 中可能传递的 alpha 或其他参数
        alpha = kwargs.get("alpha", 1.0)  # 默认值为 1.0
        beta = kwargs.get("beta", 0.5)  # 同样处理 beta 参数

        # 用 tqdm 显示训练进度条
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1} Training") as pbar:
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(
                    device
                )  # 移动数据到正确设备
                optimizer.zero_grad()  # 清除上一步的梯度
                outputs = model(inputs)

                loss = criterion(outputs, labels) * alpha  # 使用 alpha 参数调整损失函数
                loss.backward()  # 反向传播
                optimizer.step()  # 更新参数

                running_loss += loss.item()

                # 计算准确率
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # 更新进度条显示每个 mini-batch 的损失
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                pbar.update(1)

        avg_loss = running_loss / len(train_loader)  # 计算平均损失
        accuracy = correct / total  # 计算训练集的准确率
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_loss:.4f}, Training Accuracy: {accuracy * 100:.2f}%"
        )

        # 仅在最后一次保存模型，避免每个 epoch 都保存
        if not save_final_model_only or epoch == (num_epochs - 1):
            torch.save(
                model.state_dict(),
                os.path.join(
                    save_path, f"{self.model_name}_{self.dataset_name}_final.pth"
                ),
            )
            print(
                f"Final model saved to {os.path.join(save_path, f'{self.model_name}_{self.dataset_name}_final.pth')}"
            )

    def test(self, model, test_loader, condition, progress_bar=None):
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)  # 确保模型移动到正确设备

        correct = 0
        total = 0
        running_loss = 0.0
        criterion = torch.nn.CrossEntropyLoss()  # 定义损失函数

        # 用于 early stopping 机制的测试
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(
                    device
                )  # 移动数据到正确设备
                outputs = model(images)
                loss = criterion(outputs, labels)  # 计算损失
                running_loss += loss.item()  # 累加损失

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # 更新测试进度条
                if progress_bar:
                    progress_bar.update(1)

        accuracy = correct / total
        avg_loss = running_loss / len(test_loader)  # 计算平均损失
        print(f"Test Accuracy: {100 * accuracy:.2f}%, Loss: {avg_loss:.4f}")

        # 保存测试结果为 JSON 文件
        result = {"accuracy": accuracy, "loss": avg_loss}
        save_dir = os.path.join(
            "results", self.model_name, self.dataset_name, condition
        )
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "performance.json")
        with open(save_path, "w") as f:
            json.dump(result, f)

        print(f"Performance saved to {save_path}")

        return accuracy  # 返回准确率，以用于 early stopping 机制
