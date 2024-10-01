from torch import optim


def create_optimizer_scheduler(optimizer_type, parameters,
                               learning_rate=1e-2, weight_decay=5e-4,
                               step_size=20, gamma=0.5):
    # 根据用户选择的优化器初始化
    if optimizer_type == "adam":
        optimizer = optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif optimizer_type == "sgd":  # add weight_decay, 0.7/0.8
        optimizer = optim.SGD(parameters, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    return optimizer, lr_scheduler
