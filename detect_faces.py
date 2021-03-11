#import toolbox
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import camera

# creates rectangles which are resized to fit the face
def find_faces(image,model):
    """ Using facenet_models, locate faces in a given picture and create descriptor vecors
    Parameters:
    -----------
    image: Path to image file OR (X, Y, 3) numpy array of pixels
    
    Returns:
    --------
    Tuple: (cropped_faces: List of cropped faces numpy arrays (N, X, Y, 3), where N is number of identified faces
            resized_crop: cropped_faces resized to (N, 160, 160), where N is number of identified faces)
            (0, 0): Returned if no faces are found
    """
    # formats the images by manipulating the array such that it shapes the face for detection
    if type(image) == np.ndarray:
        img = image
    else:
        img = cv2.imread(image)
        img = img[:,:,::-1]

    #detects faces
    bounding_boxes, _, landmarks = model.detect(img)

    if bounding_boxes is None:
        return (None, None, 0, 0)

    for bound in bounding_boxes:
        bound[bound<0]=0

    # Cropped Face
    cropped_face = [img[int(bounding_boxes[n][1]):int(bounding_boxes[n][3]), int(bounding_boxes[n][0]):int(bounding_boxes[n][2])] for n in range(bounding_boxes.shape[0])]
    resized_crop = np.array([cv2.cvtColor(cv2.resize(img, (160, 160)), cv2.COLOR_RGB2GRAY) for img in cropped_face])
    return landmarks, bounding_boxes, cropped_face, resized_crop