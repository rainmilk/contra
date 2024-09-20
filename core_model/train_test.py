import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn


def model_train(
    train_loader,
    model,
    optimizer,
    criterion,
    alpha,
    args,
    save_path="",
    teacher_model=False,
):
    # todo opt重置
    # 训练模型并显示进度
    print(f"Training model on {args.dataset}")

    for epoch in tqdm(range(args.num_epochs), desc="Training Progress"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)  # 确保模型移动到正确的设备
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        # 用 tqdm 显示训练进度条
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1} Training") as pbar:
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(
                    device
                )  # 移动数据到正确设备
                optimizer.zero_grad()  # 清除上一步的梯度
                if teacher_model:
                    _, outputs = model(inputs)
                else:
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

        # 仅在最后一次保存模型，避免每个 epoch 都保存
        if epoch == args.num_epochs - 1:
            torch.save(
                model.state_dict(),
                os.path.join(save_path, f"{args.model}_{args.dataset}_final.pth"),
            )
            print(
                f"Final model saved to {os.path.join(save_path, f'{args.model}_{args.dataset}_final.pth')}"
            )


def model_test(dataset, data_loader, model, teacher_model=False):
    eval_results = {}
    data, labels = dataset.data, dataset.label

    if teacher_model:
        embeds, predicts = teacher_model_forward(data_loader, model)
    else:
        probs, predicts = working_model_forward(data_loader, model)

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


def working_model_forward(data_loader, model):
    model.eval()
    output_probs, output_predicts = [], []

    for i, (image, target) in enumerate(data_loader):
        image = image.cuda()

        logics = model(image)

        probs = nn.functional.softmax(logics, dim=-1)
        probs = probs.data.cpu().numpy()
        output_probs.append(probs)

        predicts = np.argmax(probs, axis=1)
        output_predicts.append(predicts)

    return np.concatenate(output_probs, axis=0), np.concatenate(output_predicts, axis=0)


def teacher_model_forward(model, test_loader):
    model.eval()
    embed_outs, outputs = [], []

    for i, (image, target) in enumerate(test_loader):
        image = image.cuda()

        embed_out, output = model(image)  # embedding out [batch, 512]

        embed_outs.append(embed_out.data.cpu().numpy())
        output = output.data.cpu().numpy()
        output = np.argmax(output, axis=1)
        outputs.append(output)

    return np.concatenate(embed_outs, axis=0), np.concatenate(outputs, axis=0)
