import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn


def model_train(
    train_loader,
    model,
    optimizer,
    lr_scheduler,
    criterion,
    alpha,
    args,
    device="cuda",
    save_path="",
):
    # todo opt 重置
    # 训练模型并显示进度
    print(f"Training model on {args.dataset}")

    model = model.to(device)  # 确保模型移动到正确的设备
    model.train()

    iters = len(train_loader)
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

        # 仅在最后一次保存模型，避免每个 epoch 都保存
        if epoch == args.num_epochs - 1:
            os.makedirs(save_path, exist_ok=True)
            torch.save(
                model.state_dict(),
                os.path.join(save_path, f"{args.model}_{args.dataset}_final.pth"),
            )
            print(
                f"Final model saved to {os.path.join(save_path, f'{args.model}_{args.dataset}_final.pth')}"
            )


def model_test(labels, data_loader, model, device="cuda", teacher_model=False):
    eval_results = {}

    if teacher_model:
        predicts, probs, embeds = teacher_model_forward(data_loader, model, device)
    else:
        predicts, probs = working_model_forward(data_loader, model, device)

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


def working_model_forward(data_loader, model, device="cuda"):
    model = model.to(device)
    model.eval()

    output_probs, output_predicts = [], []

    for i, (image, target) in enumerate(data_loader):
        image = image.to(device)
        logics = model(image)

        probs = nn.functional.softmax(logics, dim=-1)
        probs = probs.data.cpu().numpy()
        output_probs.append(probs)

        predicts = np.argmax(probs, axis=1)
        output_predicts.append(predicts)

    return np.concatenate(output_predicts, axis=0), np.concatenate(output_probs, axis=0)


def teacher_model_forward(test_loader, model, device="cuda"):
    model = model.to(device)
    model.eval()

    embed_outs, output_probs, output_predicts = [], [], []

    for i, (image, target) in enumerate(test_loader):
        image = image.to(device)  # 数据移动到设备

        embed_out, logics = model(image)  # embedding out [batch, 512]

        embed_outs.append(embed_out.data.cpu().numpy())

        probs = nn.functional.softmax(logics, dim=-1)
        probs = probs.data.cpu().numpy()
        output_probs.append(probs)

        predicts = np.argmax(probs, axis=1)
        output_predicts.append(predicts)

    return (
        np.concatenate(output_predicts, axis=0),
        np.concatenate(output_probs, axis=0),
        np.concatenate(embed_outs, axis=0),
    )
