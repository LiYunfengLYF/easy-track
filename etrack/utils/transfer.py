import torch


def speed2waitkey(speed):
    """
    Description
        trans fps to waitkey of cv2

    Params:
        speed:      fps, int

    return:
        waitkey:    int

    """
    if speed == 0:
        return 0
    else:
        return int((1 / speed) * 1000)


def img2tensor(img, device='cuda:0'):
    """
    Description
        transfer an img to a tensor
        mean: [0.485, 0.456, 0.406]
        std:  [0.229, 0.224, 0.225]

    Params:
        img:       np.array
        device:    default is 'cuda:0'

    return:
        Tensor:    torch.tensor(1,3,H,W)

    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).cuda()
    img_tensor = torch.tensor(img).to(device).float().permute((2, 0, 1)).unsqueeze(dim=0)
    return ((img_tensor / 255.0) - mean) / std  # (1,3,H,W)
