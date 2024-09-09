import torch
import os


class TrainTestUtils:
    def __init__(self, model_name, dataset_name):
        self.model_name = model_name
        self.dataset_name = dataset_name

    def create_save_path(self, condition):
        save_dir = os.path.join("models", self.model_name, self.dataset_name, condition)
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def train_and_save(
        self, model, train_loader, criterion, optimizer, save_path, num_epochs=10
    ):
        model.train()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}"
            )

        torch.save(
            model.state_dict(),
            os.path.join(save_path, f"{self.model_name}_{self.dataset_name}_final.pth"),
        )
        print(
            f"Model saved to {os.path.join(save_path, f'{self.model_name}_{self.dataset_name}_final.pth')}"
        )

    def test(self, model, test_loader):
        model.eval()
        correct = 0
        total = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Test Accuracy: {100 * correct / total}%")
