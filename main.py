import torchvision 
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


class ImageDetector():
    def __init__(self, path=False, num_classes=2):
        if not path:
            path = './weigths/modelv1.pth'
        self.model = self._load_model(path, num_classes)
        self.model.eval()

    def _load_model(self, path, num_classes=2):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        weigths = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(weigths)
        return model

    def get_bounding(self, img):
        # To Do:
        # - Crop and fix image
        # - Parse output (probability filter)
        with torch.no_grad():
            out = model(img)
        return out


if __name__ == '__main__':
    path = 'weigths/cath_agility_1.pth'
    model = ImageDetector(path=path, num_classes=4)