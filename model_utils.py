import torch
import torchvision
from torch import nn, optim
from nets import *


class ModelUtils:
    def __init__(self, model_name, num_classes):
        self.model_name = model_name
        self.num_classes = num_classes

    def create_model(self):
        if self.model_name == "resnet18":
            model = ResNet18()
        elif self.model_name == "vgg16":
            model = VGG("VGG16")
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        return model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def get_criterion_and_optimizer(self, model):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        return criterion, optimizer

    def setup_model_dataset(self, args):
        normalization_dict = {
            "cifar10": ([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
            "cifar100": ([0.5071, 0.4866, 0.4409], [0.2673, 0.2564, 0.2762]),
            "svhn": ([0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970]),
            "flowers102": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            "TinyImagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        }

        classes_dict = {
            "cifar10": 10,
            "cifar100": 100,
            "svhn": 10,
            "flowers102": 102,
            "TinyImagenet": 200,
        }

        if args.dataset not in normalization_dict:
            raise ValueError(f"Dataset {args.dataset} not supported!")

        normalization = NormalizeByChannelMeanStd(
            mean=normalization_dict[args.dataset][0],
            std=normalization_dict[args.dataset][1],
        )
        classes = classes_dict[args.dataset]

        # Setup dataset loaders based on dataset
        if args.dataset in ["cifar10", "cifar100", "svhn"]:
            train_full_loader, val_loader, _ = globals()[f"{args.dataset}_dataloaders"](
                batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers
            )
            marked_loader, _, test_loader = globals()[f"{args.dataset}_dataloaders"](
                batch_size=args.batch_size,
                data_dir=args.data,
                num_workers=args.workers,
                class_to_replace=args.class_to_replace,
                num_indexes_to_replace=args.num_indexes_to_replace,
                indexes_to_replace=args.indexes_to_replace,
                seed=args.seed,
                only_mark=True,
                shuffle=args.shuffle,
                no_aug=args.no_aug,
            )
        elif args.dataset == "flowers102":
            train_full_loader, val_loader, _ = Flowers102_dataloaders(
                batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers
            )
            marked_loader, _, test_loader = Flowers102_dataloaders(
                batch_size=args.batch_size,
                data_dir=args.data,
                num_workers=args.workers,
                class_to_replace=args.class_to_replace,
                num_indexes_to_replace=args.num_indexes_to_replace,
                indexes_to_replace=args.indexes_to_replace,
                seed=args.seed,
                only_mark=True,
                shuffle=args.shuffle,
                no_aug=args.no_aug,
            )
        elif args.dataset == "TinyImagenet":
            train_full_loader, val_loader, test_loader = TinyImageNet(
                args
            ).data_loaders(
                batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers
            )
            marked_loader, _, _ = TinyImageNet(args).data_loaders(
                batch_size=args.batch_size,
                data_dir=args.data,
                num_workers=args.workers,
                class_to_replace=args.class_to_replace,
                num_indexes_to_replace=args.num_indexes_to_replace,
                indexes_to_replace=args.indexes_to_replace,
                seed=args.seed,
                only_mark=True,
                shuffle=args.shuffle,
            )
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")

        # Setup the model
        if args.imagenet_arch:
            model = model_dict[args.arch](num_classes=classes, imagenet=True)
        elif args.arch == "swin_t":
            model = swin_t(
                window_size=4, num_classes=classes, downscaling_factors=(2, 2, 2, 1)
            )
        else:
            model = model_dict[args.arch](num_classes=classes)

        model.normalize = normalization

        if args.train_seed is None:
            args.train_seed = args.seed
        setup_seed(args.train_seed)

        print(model)
        return model, train_full_loader, val_loader, test_loader, marked_loader
