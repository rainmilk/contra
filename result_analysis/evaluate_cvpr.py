import argparse

import torch
import numpy as np
from args_paser import parse_args
from configs import settings
from core_model.custom_model import ClassifierWrapper, load_custom_model
from core_model.dataset import get_dataset_loader
from core_model.train_test import model_forward


def execute(args):
    case = settings.get_case(
        args.noise_ratio, args.noise_type
    )
    uni_name = args.uni_name
    num_classes = settings.num_classes_dict[args.dataset]

    loaded_model = load_custom_model(args.model, num_classes, load_pretrained=False)
    model_ckpt_path = settings.get_ckpt_path(args.dataset, case, args.model,
                                             model_suffix=args.model_suffix, unique_name=uni_name)
    model = ClassifierWrapper(loaded_model, num_classes)

    print(f"Loading model from {model_ckpt_path}")
    checkpoint = torch.load(model_ckpt_path)
    model.load_state_dict(checkpoint, strict=False)

    _, _, test_loader = get_dataset_loader(
        args.dataset,
        "test",
        None,
        batch_size=args.batch_size,
        shuffle=False,
    )

    results, embedding = model_test(test_loader, model)


def model_test(data_loader, model, device="cuda"):
    eval_results = {}

    predicts, probs, embedding, labels = model_forward(data_loader, model, device,
                                            output_embedding=True, output_targets=True)

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

    return eval_results, embedding



if __name__ == "__main__":
    try:
        pargs = parse_args()
        execute(pargs)
    except argparse.ArgumentError as e:
        print(f"Error parsing arguments: {e}")
