import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def show_image(img):
    plt.figure()
    _, ax = plt.subplots(1)
    ax.imshow(img)
    plt.axis('off')
    plt.show()

def plot_prediction(img, prediction, labels, tot_categories=100):
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i) for i in np.linspace(0, 1, tot_categories)]
    plt.figure()
    _, ax = plt.subplots(1)
    ax.imshow(img)

    for bbox, cat, prob in zip(prediction['boxes'], prediction['labels'],
                                prediction['scores']):  
        x1, y1, x2, y2 = bbox
        # Change
        color = colors[1] #colors[labels[cat]]
        box_h = abs(y2 - y1)
        box_w = abs(x2 - x1)
        bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, 
                                    edgecolor=color, facecolor='none')

        plt.text(x1, y1, s=f"{cat} prob={prob}", color='white',
                    verticalalignment='top', bbox={'color': color, 'pad': 0})

        ax.add_patch(bbox)

    plt.axis('off')
    plt.show()