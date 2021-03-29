# ## Example
# 
# In this simple example, we load an image, pre-process it, and classify it with a pretrained EfficientNet.

import json
from PIL import Image

import torch
from torchvision import transforms

from model import EfficientNet, get_image_size, from_pretrained

model_names = ['efficientnet-b%d' % i for i in range(8)]
# print(model_names)

for model_name in model_names:
    image_size = get_image_size(model_name) # 224

    # Open image
    img = Image.open('testdata/cat.jpeg')
    img

    # Preprocess image
    tfms = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size), 
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
    img = tfms(img).unsqueeze(0)

    # Load class names
    labels_map = json.load(open('testdata/labels_map.txt'))
    labels_map = [labels_map[str(i)] for i in range(1000)]

    # Classify with EfficientNet
    model = from_pretrained(model_name)
    model.eval()
    with torch.no_grad():
        logits = model(img)
    preds = torch.topk(logits, k=5).indices.squeeze(0).tolist()

    print('-----')
    for idx in preds:
        label = labels_map[idx]
        prob = torch.softmax(logits, dim=1)[0, idx].item()
        print('{:<75} ({:.2f}%)'.format(label, prob*100))