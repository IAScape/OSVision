import torchvision 
from torchvision import transforms, models
from .helpers import _show_image, _plot_prediction
import torch
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

FastRCNNPredictor = models.detection.faster_rcnn.FastRCNNPredictor

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

    @staticmethod
    def _load_model(path, num_classes=2):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        weigths = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(weigths)
        return model

    @staticmethod
    def show_image(img):
        _show_image(img)

    @staticmethod
    def plot_prediction(img, prediction):
        labels_inv = {v: k for (k, v) in labels.items()}
        _plot_prediction(img, prediction, labels_inv, 4)

    def get_bounding(self, img, threshold=0.5):
        # To Do:
        # - Parse output (probability filter)
        x_scale = 221 / img.size[0]
        y_scale = 221 / img.size[1]
        img = img.resize((221, 221))
        img = self.transform(img)

        img = img.unsqueeze(0)
        img = torch.Tensor(img)

        with torch.no_grad():
            prediction = self.model(img)[0]
    
        filtered_pred = {'boxes': [], 'labels': [], 'scores': []}
        for box, cat, prob in zip(prediction['boxes'], prediction['labels'],
                                prediction['scores']):
            if prob > threshold:
                box_scaled = [box[0].item() / x_scale, box[1].item() / y_scale, box[2].item() / x_scale, box[3].item() / y_scale]
                filtered_pred['boxes'].append(box_scaled)
                filtered_pred['labels'].append(cat)
                filtered_pred['scores'].append(round(float(prob), 3))
        return filtered_pred


if __name__ == '__main__':
    path = 'weigths/cath_agility_1.pth'
    model = ImageDetector(path=path, num_classes=4)