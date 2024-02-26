import cv2 
from PIL import Image
from ultralytics import YOLO
from loguru import logger

class BoundingBoxDetection:
    def __init__(self, 
                 model_weights_path = '../models/yolov8_trained.pt') -> None:
        """
        Initialize the YOLO model with the given weights
        Args:
            model_weights_path: str - path to the model weights
        """
        self.model = YOLO(model_weights_path)
    
    def detect(self, 
               image_path,
               imgsz = 640,
               save_crop_images = True) -> None:
        """
        Detect the bounding boxes in the given image
        Args:
            image_path: str - path to the image
            imgsz: int - size of the image
            save_crop_images: bool - whether to save the cropped images
        """
        self.results = self.model([image_path],  imgsz = imgsz, save_crop = save_crop_images, project = '../results')
        logger.success('Bounding box detection completed')
        
    def save_results(self, 
                     save_path = '../results/bounding_box_detection.jpg') -> None:
        """
        Save the results of the bounding box detection
        Args:
            save_path: str - path to save the image
        """
        
        for r in self.results:
            im_array = r.plot()  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            im.save(save_path)  # save image

if __name__ == '__main__':
    image_path = '../data/IMG_0105.jpg'
    bbd = BoundingBoxDetection()
    bbd.detect(image_path)
    bbd.save_results()