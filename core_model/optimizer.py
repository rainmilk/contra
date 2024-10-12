from torch import optim


def create_optimizer_scheduler(
    optimizer_type,
    parameters,
    epochs,
    learning_rate=1e-2,
    weight_decay=1e-3,
    gamma=0,
    min_epochs_for_decay=10,
    factor=0.99
):
    # 根据用户选择的优化器初始化
    if optimizer_type == "adam":
        optimizer = optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == "sgd":  # add weight_decay, 0.7/0.8
        optimizer = optim.SGD(parameters, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    if epochs >= min_epochs_for_decay:
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=gamma
        )
    else:
        lr_scheduler = optim.lr_scheduler.ConstantLR(
            optimizer, factor=factor, total_iters=epochs
        )

    return optimizer, lr_scheduler
