import torch
import torchvision.transforms.functional as TF


def tensor_rot_90(x):
    x = TF.rotate(x, 90)
    return x


def tensor_rot_180(x):
    x = TF.rotate(x, 180)
    return x


def tensor_rot_270(x):
    x = TF.rotate(x, 270)
    return x


def rotate_batch_with_labels(batch, labels):
    images = []
    for img, label in zip(batch, labels):
        if label == 1:
            img = tensor_rot_90(img)
        elif label == 2:
            img = tensor_rot_180(img)
        elif label == 3:
            img = tensor_rot_270(img)
        images.append(img.unsqueeze(0))
    return torch.cat(images)


def rotate_batch(batch, label):
    if label == 'rand':
        labels = torch.randint(4, (len(batch),), dtype=torch.long)
    elif label == 'expand':
        labels = torch.cat([torch.zeros(len(batch), dtype=torch.long),
                            torch.zeros(len(batch), dtype=torch.long) + 1,
                            torch.zeros(len(batch), dtype=torch.long) + 2,
                            torch.zeros(len(batch), dtype=torch.long) + 3])
        batch = batch.repeat((4, 1, 1, 1))
    else:
        assert isinstance(label, int)
        labels = torch.zeros((len(batch),), dtype=torch.long) + label
    return rotate_batch_with_labels(batch, labels), labels