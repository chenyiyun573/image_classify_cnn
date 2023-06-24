import torch

dataset_path = '/home/superbench/v-yiyunchen/net/'

batch_size = 512

from torchvision.models import resnet18

epoch_c = 30

def get_model_and_device(model_name):
    # Pre-trained results
    
    if model_name == "0":
        from torchvision.models import alexnet
        model = alexnet(pretrained=True)
    elif model_name == "1":
        from torchvision.models import vgg19
        model = vgg19(pretrained=True)
    elif model_name == "2":
        from torchvision.models import resnet18
        model = resnet18(pretrained=True)
    elif model_name == "3":
        from torchvision.models import resnet152
        model = resnet152(pretrained=True)
    elif model_name == "4":
        from torchvision.models import densenet201
        model = densenet201(pretrained=True)
    elif model_name == "5":
        from torchvision.models import inception_v3
        model = inception_v3(pretrained=True)
    elif model_name == "6":
        from torchvision.models import googlenet
        model = googlenet(pretrained=True)
    elif model_name == "7":
        from torchvision.models import squeezenet1_1
        model = squeezenet1_1(pretrained=True)
    else:
        raise ValueError("Invalid model name")
    
    # Non pretrained
    """
    if model_name == "0":
        from torchvision.models import alexnet
        model = alexnet(pretrained=False)
    elif model_name == "1":
        from torchvision.models import vgg19
        model = vgg19(pretrained=False)
    elif model_name == "2":
        from torchvision.models import resnet18
        model = resnet18(pretrained=False)
    elif model_name == "3":
        from torchvision.models import resnet152
        model = resnet152(pretrained=False)
    elif model_name == "4":
        from torchvision.models import densenet201
        model = densenet201(pretrained=False)
    elif model_name == "5":
        from torchvision.models import inception_v3
        model = inception_v3(pretrained=False)
    elif model_name == "6":
        from torchvision.models import googlenet
        model = googlenet(pretrained=False)
    elif model_name == "7":
        from torchvision.models import squeezenet1_1
        model = squeezenet1_1(pretrained=False)
    else:
        raise ValueError("Invalid model name")
    """
    # Add more models here with elif statements
    device = torch.device(f"cuda:{model_name}" if torch.cuda.is_available() else "cpu")
    return model, device

