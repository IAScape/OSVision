import torchvision 
from torchvision import transforms
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

labels = {
    1: 'tag',
    2: 'tagred',
    3: 'mark'
}

class ImageDetector():
    def __init__(self, path=False, num_classes=2):
        if not path:
            path = './weigths/modelv1.pth'
        self.model = self._load_model(path, num_classes)
        self.model.eval()
        self.transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])

    def _load_model(self, path, num_classes=2):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        weigths = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(weigths)
        return model

    def get_bounding(self, img, threshold=0.5):
        # To Do:
        # - Parse output (probability filter)
        x_scale = 221 / img.size[0]
        y_scale = 221 / img.size[1]
        img = img.resize((221, 221))
        img = self.transform(img)

        with torch.no_grad():
            out = self.model([img])[0]
        filtered_out = []
        for box, label, score in zip(out['boxes'], out['labels'], out['scores']):
            if score > threshold:
                inv_x = 1 / x_scale
                inv_y = 1 / y_scale
                box = [box[0] * inv_x, box[1] * inv_y, box[2] * inv_x, box[3] * inv_y]
                filtered_out.append((labels[label.item()], box, score))
        return filtered_out


if __name__ == '__main__':
    path = 'weigths/cath_agility_1.pth'
    model = ImageDetector(path=path, num_classes=4)