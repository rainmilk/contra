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
    alpha=1,
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

    model = model.to(device)  # 确保模型移动到正确的设备
    model.train()

    if mix_classes > 0:
        cutmix_transform = v2.CutMix(alpha=1.0, num_classes=mix_classes)
        mixup_transform = v2.MixUp(alpha=0.5, num_classes=mix_classes)

    for epoch in tqdm(range(args.num_epochs), desc="Training Progress"):
        running_loss = 0.0
        correct = 0
        total = 0

        # 更新学习率调度器
        lr_scheduler.step(epoch)

        # 用 tqdm 显示训练进度条
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1} Training") as pbar:
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                if mix_classes > 0:
                    transform = np.random.choice([cutmix_transform, mixup_transform])
                    labels = labels.to(torch.long)
                    inputs, labels = transform(inputs, labels)

                optimizer.zero_grad()  # 清除上一步的梯度
                outputs = model(inputs)

                loss = criterion(outputs, labels) * alpha  # 使用 alpha 参数调整损失函数
                loss.backward()  # 反向传播
                optimizer.step()  # 更新参数
                running_loss += loss.item()

                # 计算准确率
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                labels_ = torch.argmax(labels, dim=-1)
                correct += (predicted == labels_).sum().item()

                # 更新进度条显示每个 mini-batch 的损失
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                pbar.update(1)

        avg_loss = running_loss / len(train_loader)  # 计算平均损失
        accuracy = correct / total  # 计算训练集的准确率
        print(
            f"Epoch [{epoch + 1}/{args.num_epochs}], Training Loss: {avg_loss:.4f}, Training Accuracy: {accuracy * 100:.2f}%"
        )
        torch.cuda.empty_cache()

        if test_loader is not None and epoch % test_per_it == 0:
            model_test(test_loader, model, device)

        # 仅在最后一次保存模型，避免每个 epoch 都保存
        if epoch == args.num_epochs - 1:
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
    model = model.to(device)
    model.eval()

    output_probs, output_predicts = [], []
    if output_embedding:
        embed_outs = []

    if output_targets:
        targets = []

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

