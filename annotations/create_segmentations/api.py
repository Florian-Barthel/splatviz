import torch
import torchvision.transforms as transforms

from model import BiSeNet

cp = 'annotations/create_segmentations/79999_iter.pth'
net = BiSeNet(n_classes=19).cuda()
net.load_state_dict(torch.load(cp))
net.eval()
normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def evaluate(image_tensor):
    img = normalize(image_tensor)
    img = torch.unsqueeze(img, 0)
    img = torch.nn.functional.interpolate(img, size=(512, 512))
    return net(img.cuda())[0]




