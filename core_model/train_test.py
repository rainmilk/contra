import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import v2


def model_train(
    train_loader,
    model,
    optimizer,
    lr_scheduler,
    criterion,
    epochs=5,
    args=None,
    device="cuda",
    save_path=".",
    mix_classes=0,
    test_loader=None,
    test_per_it=1
):
    # todo opt 重置
    # 训练模型并显示进度
    print(f"Training model on {args.dataset}")

    # model = model.to(device)  # 确保模型移动到正确的设备
    model.train()
    criterion = nn.CrossEntropyLoss()

    # 用于存储训练和测试的损失和准确率
    train_losses = []
    test_accuracies = []

    use_data_aug = mix_classes > 0

    if use_data_aug:
        cutmix_transform = v2.CutMix(alpha=1.0, num_classes=mix_classes)
        mixup_transform = v2.MixUp(alpha=0.5, num_classes=mix_classes)

    for epoch in tqdm(range(epochs), desc="Training Progress"):
        running_loss = 0.0
        correct = 0
        total = 0

        # 更新学习率调度器
        lr_scheduler.step(epoch)

        # tqdm 进度条显示
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1} Training") as pbar:
            for inputs, targets in train_loader:
                if use_data_aug:
                    transform = np.random.choice([mixup_transform, cutmix_transform])
                    targets = targets.to(torch.long)
                    inputs, targets = transform(inputs, targets)

                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                mixed_max = torch.argmax(targets.data, 1) if use_data_aug else targets
                total += targets.size(0)
                correct += (predicted == mixed_max).sum().item()

                # 更新进度条
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                pbar.update(1)

        # 打印训练集的平均损失和准确率
        avg_loss = running_loss / len(train_loader)
        accuracy = correct / total
        train_losses.append(avg_loss)
        print(
            f"Epoch [{epoch + 1}/{epochs}], Training Loss: {avg_loss:.4f}, Training Accuracy: {accuracy * 100:.2f}%"
        )

        model_test_global(model, test_loader, epoch, device)
        # if test_loader is not None and epoch % test_per_it == 0:
        #     model_test_global(model, test_loader, epoch, device)

        # 仅在最后一次保存模型，避免每个 epoch 都保存
        if epoch == epochs - 1:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(
                f"Model has saved to {save_path}."
            )


def model_test(data_loader, model, device="cuda"):
    eval_results = {}

    predicts, probs, labels = model_forward(data_loader, model, device, output_targets=True)

    # global acc
    global_acc = np.mean(predicts == labels)
    print("test_acc: %.2f" % (global_acc * 100))
    eval_results["global"] = global_acc.item()

    # class acc
    label_list = sorted(list(set(labels)))
    for label in label_list:
        cls_index = labels == label
        class_acc = np.mean(predicts[cls_index] == labels[cls_index])
        print("label: %s, acc: %.2f" % (label, class_acc * 100))
        eval_results["label_" + str(label.item())] = class_acc.item()

    return eval_results


def model_forward(test_loader, model, device="cuda",
                          output_embedding=False, output_targets=False):
    model.eval()

    output_probs, output_predicts = [], []
    if output_embedding:
        embed_outs = []

    if output_targets:
        targets = []

    with torch.no_grad():
        for i, (image, target) in enumerate(test_loader):
            image = image.to(device)  # 数据移动到设备

            if output_embedding:
                logics, embed_out = model(image, output_embedding)
                embed_outs.append(embed_out.data.cpu().numpy())
            else:
                logics = model(image)

            probs = nn.functional.softmax(logics, dim=-1)
            probs = probs.data.cpu().numpy()
            output_probs.append(probs)

            predicts = np.argmax(probs, axis=1)
            output_predicts.append(predicts)

            if output_targets:
                targets.append(target.data.cpu().numpy())

    output_predicts = np.concatenate(output_predicts, axis=0)
    output_probs = np.concatenate(output_probs, axis=0)

    ret = [output_predicts, output_probs]

    if output_embedding:
        ret.append(np.concatenate(embed_outs, axis=0))

    if output_targets:
        ret.append(np.concatenate(targets, axis=0))

    return tuple(ret)


def model_test_global(model, test_loader, epoch, device="cuda"):
    model.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        with tqdm(
                total=len(test_loader), desc=f"Epoch {epoch + 1} Testing"
        ) as pbar:
            for test_inputs, test_targets in test_loader:
                test_inputs, test_targets = test_inputs.to(device), test_targets.to(
                    device
                )
                test_outputs = model(test_inputs)
                _, predicted_test = torch.max(test_outputs, 1)
                total_test += test_targets.size(0)
                correct_test += (predicted_test == test_targets).sum().item()

                # 更新进度条
                pbar.update(1)

    test_loss /= len(test_loader)
    test_accuracy = 100 * correct_test / total_test
    print(f"Test Accuracy after Epoch {epoch + 1}: {test_accuracy:.2f}%")

