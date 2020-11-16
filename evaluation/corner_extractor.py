''' Document Localization using Recursive CNN
 Maintainer : Khurram Javed
 Email : kjaved@ualberta.ca '''

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import cv2

import model

class GetCorners:
    def __init__(self, checkpoint_dir):
        self.model = model.ModelFactory.get_model("resnet", 'document')
        self.model.load_state_dict(torch.load(checkpoint_dir, map_location='cpu'))
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

    def get(self, pil_image):

        with torch.no_grad():
            image_array = np.copy(pil_image)
            pil_image = Image.fromarray(pil_image)
            test_transform = transforms.Compose([transforms.Resize([32, 32]),
                                                 transforms.ToTensor()])
            img_temp = test_transform(pil_image)

            img_temp = img_temp.unsqueeze(0)
            if torch.cuda.is_available():
                img_temp = img_temp.cuda()

            model_prediction = self.model(img_temp).cpu().data.numpy()[0]

            model_prediction = np.array(model_prediction)

            x_cords = model_prediction[[0, 2, 4, 6]]
            y_cords = model_prediction[[1, 3, 5, 7]]

            x_cords = x_cords * image_array.shape[1]
            y_cords = y_cords * image_array.shape[0]

            return [(x, y) for x, y in zip(x_cords, y_cords)]
