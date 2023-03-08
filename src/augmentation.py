import json
from typing import Tuple, Any
import os
import numpy
import cv2
import random
import numpy as np
import albumentations as A
from numpy import ndarray
from Utils import DataUtils
from inspect import signature


class ClassificationAugmentations:

    @staticmethod
    def resize(image: numpy.ndarray, width: int, height: int) -> numpy.ndarray:
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def random_erase(image: numpy.ndarray, max_erase: float, probability: float) -> numpy.ndarray:
        if random.uniform(0, 1) <= probability:
            max_width = image[0].__len__()
            max_height = image.__len__()
            crop_width = random.randint(0, int(max_width * max_erase))
            crop_height = random.randint(0, int(max_height * max_erase))
            crop_x0 = random.randint(0, max_width - crop_width)
            crop_y0 = random.randint(0, max_height - crop_height)
            crop_x1 = crop_x0 + crop_width
            crop_y1 = crop_y0 + crop_height
            for i in range(np.shape(image)[2]):
                image[crop_y0: crop_y1, crop_x0: crop_x1, i] = random.randint(0, 255)
        return image

    @staticmethod
    def random_horizontal_flip(image: numpy.ndarray, probability: float) -> numpy.ndarray:
        if random.uniform(0, 1) <= probability:
            return cv2.flip(image, 1)
        return image

    @staticmethod
    def random_vertical_flip(image: numpy.ndarray, probability: float) -> numpy.ndarray:
        if random.uniform(0, 1) <= probability:
            image = cv2.flip(image, 0)
        return image

    @staticmethod
    def random_crop(image: numpy.ndarray, probability: float, min_crop: float) -> numpy.ndarray:
        if random.uniform(0, 1) <= probability:
            width = image[0].__len__()
            height = image.__len__()
            if not (0 <= min_crop <= 1):
                min_crop = 0.3
            factor = random.uniform(min_crop, 1.0)
            crop_width = int(width * factor)
            crop_height = int(height * factor)
            x0 = random.randint(0, width - crop_width)
            y0 = random.randint(0, height - crop_height)
            crop = image[y0: (y0 + crop_height), x0: (x0 + crop_width)]
            image = cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)
        return image

    @staticmethod
    def random_brightness(image: np.ndarray, min_beta: int, max_beta: int) -> np.ndarray:
        if not (-127 <= min_beta <= 127):
            min_beta = -100
        if not (-127 <= max_beta <= 127):
            min_beta = 100
        return cv2.convertScaleAbs(image, beta=random.randint(min_beta, max_beta))

    @staticmethod
    def random_rotation(image: np.ndarray, probability: float, min_degree: int, max_degree: int) -> np.ndarray:
        if random.uniform(0, 1) <= probability:
            height, width = image.shape[:2]
            center = (width / 2, height / 2)
            rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=random.randint(min_degree, max_degree),
                                                    scale=1)
            image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))
        return image

    @staticmethod
    def random_hue_saturation_value(image: np.ndarray, probability: float) -> np.ndarray:
        if random.uniform(0, 1) <= probability:
            transforms = A.Compose([
                A.augmentations.transforms.HueSaturationValue(hue_shift_limit=20,
                                                              sat_shift_limit=50,
                                                              val_shift_limit=20, p=probability)
            ])
            image = transforms(image=image)['image']
        return image

    @staticmethod
    def random_gauss_noise(image: np.ndarray, var_min: int, var_max: int, mean: int, probability: float) -> np.ndarray:
        transforms = A.Compose([
            A.GaussNoise(var_limit=(var_min, var_max), mean=mean, p=probability)
        ])
        return transforms(image=image)['image']


'''
This class offers augmentations specific for the generation of object detection datasets.
The main difference to ClassificationAugmentations is that its augmentation methods adjust box coordinates too.
'''


class DetectionAugmentations:

    @staticmethod
    def resize(image: np.ndarray, boxes: list, width: int, height: int) -> tuple:
        aug_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        factor_x = width / image[0].__len__()
        factor_y = height / image.__len__()
        aug_boxes = []
        for box in boxes:
            aug_boxes.append({
                'x0': int(box['x0'] * factor_x),
                'y0': int(box['y0'] * factor_y),
                'x1': int(box['x1'] * factor_x),
                'y1': int(box['y1'] * factor_y)
            })
        return aug_image, aug_boxes

    @staticmethod
    def overlay(image: np.ndarray, obj: np.ndarray) -> tuple:
        obj = DetectionAugmentations.object_resize(image, obj)
        max_width = image[0].__len__() - obj[0].__len__()
        max_height = image.__len__() - obj.__len__()
        start_width = random.randint(0, max_width)
        start_height = random.randint(0, max_height)
        for i in range(obj.__len__()):
            for j in range(obj[0].__len__()):
                if obj[i][j][3] != 0:
                    image[start_height + i][start_width + j][0] = obj[i][j][0]
                    image[start_height + i][start_width + j][1] = obj[i][j][1]
                    image[start_height + i][start_width + j][2] = obj[i][j][2]
        box = {
            'x0': start_width,
            'y0': start_height,
            'x1': start_width + obj[0].__len__(),
            'y1': start_height + obj.__len__()
        }
        return image, box

    @staticmethod
    def object_resize(image: np.ndarray, obj: np.ndarray) -> np.ndarray:
        object_width = obj[0].__len__()
        object_height = obj.__len__()
        background_width = image[0].__len__()
        background_height = image.__len__()
        min_object_width = int(min([background_width, background_height]) / 12)
        max_object_width = int(min([background_width, background_height]) / 4)
        random_width = random.randint(min_object_width, max_object_width)
        factor = random_width / max([object_width, object_height])
        if object_width >= object_height:
            new_object_width = random_width
            new_object_height = object_height * factor
        else:
            new_object_height = random_width
            new_object_width = object_width * factor
        return cv2.resize(obj, (int(new_object_width), int(new_object_height)))

    @staticmethod
    def random_object_erase(image: np.ndarray, boxes: list, max_erase: float, probability: float) -> tuple:
        for box in boxes:
            image[box['y0']:box['y1'], box['x0']:box['x1']] = ClassificationAugmentations.random_erase(
                image[box['y0']:box['y1'], box['x0']:box['x1']], max_erase, probability)
        return image, boxes

    @staticmethod
    def random_horizontal_flip(image: np.ndarray, boxes: list, probability: float) -> tuple:
        if random.uniform(0, 1) <= probability:
            flipped_boxes = []
            image = cv2.flip(image, 1)
            width = image[0].__len__()
            for box in boxes:
                flipped_boxes.append({
                    'x0': width - int(box['x1']),
                    'y0': box['y0'],
                    'x1': width - int(box['x0']),
                    'y1': box['y1']
                }
                )
            boxes = flipped_boxes
        return image, boxes

    @staticmethod
    def random_vertical_flip(image: np.ndarray, boxes: list, probability: float) -> tuple:
        if random.uniform(0, 1) <= probability:
            flipped_boxes = []
            image = cv2.flip(image, 0)
            height = image.__len__()
            for box in boxes:
                flipped_boxes.append({
                    'x0': box['x0'],
                    'y0': height - int(box['y1']),
                    'x1': box['x1'],
                    'y1': height - int(box['y0'])
                }
                )
            boxes = flipped_boxes
        return image, boxes


class AugmentApplier:
    GENERAL_AUGMENTS = {
        "random_brightness": ClassificationAugmentations.random_brightness,
        "random_hue_saturation_value": ClassificationAugmentations.random_hue_saturation_value,
        "random_gauss_noise": ClassificationAugmentations.random_gauss_noise
    }

    CLASSIFICATION_AUGMENTS = {
        "resize": ClassificationAugmentations.resize,
        "random_crop": ClassificationAugmentations.random_crop,
        "random_horizontal_flip": ClassificationAugmentations.random_horizontal_flip,
        "random_vertical_flip": ClassificationAugmentations.random_vertical_flip,
        "random_rotation": ClassificationAugmentations.random_rotation,
        "random_erase": ClassificationAugmentations.random_erase
    }

    DETECTION_AUGMENTS = {
        "resize": DetectionAugmentations.resize,
        "random_horizontal_flip": DetectionAugmentations.random_horizontal_flip,
        "random_vertical_flip": DetectionAugmentations.random_vertical_flip,
        "random_object_erase": DetectionAugmentations.random_object_erase
    }

    @staticmethod
    def validate_config(config: str) -> bool:
        if not os.path.isfile(config):
            print("Could not find config file under specified path..")
            return False
        if not DataUtils.is_file_json(config):
            print("Config file is not a valid json.")
            return False
        f = open(config, 'r')
        augments = json.load(f)
        for k, v in augments.items():
            if type(v) is not list:
                print("Invalid arguments found, must be of type list: " + str(k) + ": " + str(v))
                return False
            elif k in AugmentApplier.GENERAL_AUGMENTS:
                if len(v) != (len(signature(AugmentApplier.GENERAL_AUGMENTS[k]).parameters) - 1):
                    print("Wrong number of arguments found for following augmentation: " + str(k))
                    return False
            elif k in AugmentApplier.CLASSIFICATION_AUGMENTS:
                if len(v) != (len(signature(AugmentApplier.CLASSIFICATION_AUGMENTS[k]).parameters) - 1):
                    print("Wrong number of arguments found for following augmentation: " + str(k))
                    return False
            elif k in AugmentApplier.DETECTION_AUGMENTS:
                if len(v) != (len(signature(AugmentApplier.DETECTION_AUGMENTS[k]).parameters) - 2):
                    print("Wrong number of arguments found for following augmentation: " + str(k))
                    return False
            else:
                print("Found invalid augmentation: " + str(k))
                return False
        return True

    @staticmethod
    def apply(image: np.ndarray, config: str, boxes: list = None) -> tuple[ndarray | Any, list | None | Any]:
        with open(config, 'r') as f:
            augments = json.load(f)
            aug_boxes = boxes
            for augmentation, args in augments.items():
                if augmentation in AugmentApplier.GENERAL_AUGMENTS:
                    image = AugmentApplier.GENERAL_AUGMENTS[augmentation](image, *args)
                elif augmentation in AugmentApplier.CLASSIFICATION_AUGMENTS and boxes is None:
                    image = AugmentApplier.CLASSIFICATION_AUGMENTS[augmentation](image, *args)
                elif augmentation in AugmentApplier.DETECTION_AUGMENTS and boxes is not None:
                    image, aug_boxes = AugmentApplier.DETECTION_AUGMENTS[augmentation](image, aug_boxes, *args)
            return image, aug_boxes
