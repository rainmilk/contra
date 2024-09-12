from .cifar-10/vgg import *
from .cifar-10/dpn import *
from .cifar-10/lenet import *
from .cifar-10/senet import *
from .cifar-10/pnasnet import *
from .cifar-10/densenet import *
from .cifar-10/googlenet import *
from .cifar-10/shufflenet import *
from .cifar-10/shufflenetv2 import *
from .cifar-10/resnet import *
from .cifar-10/resnext import *
from .cifar-10/preact_resnet import *
from .cifar-10/mobilenet import *
from .cifar-10/mobilenetv2 import *
from .cifar-10/efficientnet import *
from .cifar-10/regnet import *
from .cifar-10/dla_simple import *
from .cifar-10/dla import *


from .ResNet import *
from .ResNets import *
from .swin import *
from .VGG import *
from .VGG_LTH import *

model_dict = {
    "resnet18": resnet18,
    "resnet50": resnet50,
    "resnet20s": resnet20s,
    "resnet44s": resnet44s,
    "resnet56s": resnet56s,
    "vgg16_bn": vgg16_bn,
    "vgg16_bn_lth": vgg16_bn_lth,
}
