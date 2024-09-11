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
    ):
        """
        :param save_final_model_only: If True, only save the model after the final epoch.
        """
        model.train()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        running_loss = 0.0

        # 用 tqdm 显示训练进度条
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1} Training") as pbar:
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                # 更新进度条
                pbar.set_postfix({"Loss": loss.item()})
                pbar.update(1)

        print(f"Epoch [{epoch + 1}], Loss: {running_loss / len(train_loader)}")

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
        correct = 0
        total = 0
        running_loss = 0.0
        criterion = torch.nn.CrossEntropyLoss()  # 定义损失函数
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 用于 early stopping 机制的测试
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
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
