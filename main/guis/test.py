import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from classification.ImplantClassify_v3 import ImageClassificationModel
from PIL import Image

classify_model = ImageClassificationModel(data_dir=None, num_epochs=None)
classify_model = classify_model.load("./best_model/best_classification_model.pth")

image_path = './guis/test.jpg'
image = Image.open(image_path)
prediction = classify_model.predict(image)
