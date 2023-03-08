import os
import random
import types
from abc import abstractmethod
import cv2
from augmentation import DetectionAugmentations, AugmentApplier
from annotation_writers import DetectionAnnotations
from custom_exceptions import InvalidGeneratorArgumentException


class Generator:

    def __init__(self, image_source: str, destination: str, num: int, augment_config: str):
        self.image_source = None
        self.set_image_source(image_source)
        self.set_destination(destination)
        self.set_num(num)
        self.set_augment_config(augment_config)

        # check if a directory exists and whether it is empty

    @staticmethod
    def valid_dir(dir: str) -> bool:
        if os.path.isdir(dir) and len(os.listdir(dir)) != 0:
            return True
        return False

    @staticmethod
    def dir_contains_only_imgs(dir: str) -> bool:
        for file in os.listdir(dir):
            img = cv2.imread(os.path.join(dir, file))
            # if img is of type NoneType then it cannot be a valid image file (expected: numpy.ndarray)
            if type(img) == types.NoneType:
                return False
        return True

    @staticmethod
    def dir_contains_only_pngs(dir: str) -> bool:
        if Generator.dir_contains_only_imgs(dir):
            for file in os.listdir(dir):
                img = cv2.imread(os.path.join(dir, file), cv2.IMREAD_UNCHANGED)
                # real PNG images have a shape of (x,x,4)
                if img.shape[2] != 4:
                    return False
            return True
        return False

    @staticmethod
    def get_random_file(dir: str) -> str:
        files = os.listdir(dir)
        return os.path.join(dir, files[random.randint(0, len(files) - 1)])

    def set_image_source(self, image_source: str) -> bool:
        if not Generator.valid_dir(image_source):
            raise InvalidGeneratorArgumentException("Image source directory is not a directory or is empty.")
        if not Generator.dir_contains_only_imgs(image_source):
            raise InvalidGeneratorArgumentException("Image directory must only contain images (jpeg, png, ...).")
        self.image_source = image_source

    def set_destination(self, destination: str) -> bool:
        if not os.path.isdir(destination):
            raise InvalidGeneratorArgumentException("No directory found under destination path.")
        self.destination = destination

    def set_num(self, num: int) -> bool:
        if type(num) == int and num < 0:
            raise InvalidGeneratorArgumentException("Number of set output images must be positive.")
        if type(num) is None:
            self.num = None
        else:
            self.num = num

    def set_augment_config(self, augment_config: str) -> bool:
        if not AugmentApplier.validate_config(augment_config):
            raise InvalidGeneratorArgumentException("Invalid config file.")
        self.augment_config = augment_config

    @abstractmethod
    def generate(self):
        pass


class ClassificationGenerator(Generator):

    def __init__(self, image_source: str, destination: str, num: int, augment_config: str):
        super().__init__(image_source, destination, num, augment_config)

    def generate(self):
        samples = []
        if self.num is None:
            for file in os.listdir(self.image_source):
                samples.append(os.path.join(self.image_source, file))
        else:
            for i in range(self.num):
                samples.append(Generator.get_random_file(self.image_source))
        for idx, item in enumerate(samples):
            image = cv2.imread(item)
            image, _ = AugmentApplier.apply(image, self.augment_config)
            file_name = os.path.basename(os.path.normpath(self.destination)) + "_" + str(idx) + ".jpg"
            cv2.imwrite(os.path.join(self.destination, file_name), image)
            print(file_name)
        print("Generating finished.")


'''
This class enables user to synthesize datasets for object detection training & testing.
It does so by randomly overlapping "object" images onto "background" images.
Additionally the resulting images will be augmented and annotated in different formats.
'''


class DetectionGenerator(Generator):
    object_source = None
    annotation_format = None
    annotation_output_dir = None
    object_source = None
    annotation_format = None
    annotation_output_dir = None

    def __init__(self, image_source: str, object_source: str, num: int, destination: str, annotation_format: str,
                 annotation_output_dir: str, augment_config: str):
        super().__init__(image_source, destination, num, augment_config)
        self.set_objects_source(object_source)
        self.set_annotation_format(annotation_format)
        self.set_annotation_output_dir(annotation_output_dir)

    def set_objects_source(self, objects_source: str) -> bool:
        if not Generator.valid_dir(objects_source):
            raise InvalidGeneratorArgumentException("Object source directory not a real directory or is empty.")
        for dir in os.listdir(objects_source):
            path = os.path.join(objects_source, dir)
            if not Generator.valid_dir(path):
                raise InvalidGeneratorArgumentException("Found invalid or empty sub-directory in object source.")
            if not Generator.dir_contains_only_pngs(path):
                raise InvalidGeneratorArgumentException("Found invalid png image in object source sub-directory.")
        self.objects_source = objects_source

    def set_annotation_format(self, annotation_format) -> bool:
        if annotation_format is None:
            self.annotation_format = 'xml'
        else:
            if not DetectionAnnotations.accepted_format(annotation_format):
                raise InvalidGeneratorArgumentException("Specified annotation format not a valid format, accepted formats are xml & json")
            self.annotation_format = annotation_format

    def set_annotation_output_dir(self, output_dir: str) -> bool:
        if not os.path.isdir(output_dir):
            raise InvalidGeneratorArgumentException("No directory found under specified annotation-output-path.")
        if output_dir[len(output_dir) - 1] != "/":
            self.annotation_output_dir = output_dir + "/"
        else:
            self.annotation_output_dir = output_dir

    def get_random_objects(self, max_objects: int) -> tuple:
        objects = []
        labels = []
        dir_list = os.listdir(self.objects_source)
        for i in range(random.randint(1, max_objects)):
            idx = random.randint(0, len(dir_list) - 1)
            class_dir = os.path.join(self.objects_source, dir_list[idx])
            images = os.listdir(class_dir)
            objects.append(os.path.join(class_dir, images[random.randint(0, len(images) - 1)]))
            labels.append(dir_list[idx])
        return objects, labels

    def create_image(self, background_path, objects, img_idx) -> tuple:
        boxes = []
        background = cv2.imread(background_path)
        # copy paster objects onto background image
        for obj in objects:
            obj_img = cv2.imread(obj, cv2.IMREAD_UNCHANGED)
            background, box = DetectionAugmentations.overlay(background, obj_img)
            boxes.append(box)
        # some image augmentations
        background, boxes = AugmentApplier.apply(background, self.augment_config, boxes)
        # remaining image annotations
        img_path = self.destination + str(img_idx) + ".jpg"
        img_width = background[0].__len__()
        img_height = background.__len__()
        cv2.imwrite(img_path, background)
        return img_path, img_width, img_height, boxes

    def generate(self):
        for i in range(self.num):
            background_img = Generator.get_random_file(self.image_source)
            object_imgs, labels = self.get_random_objects(7)
            img_path, img_width, img_height, boxes = self.create_image(background_img, object_imgs, i)
            DetectionAnnotations.write_annotations(self.annotation_format, img_path, img_width, img_height, labels,
                                                   boxes, self.annotation_output_dir)
            print(img_path)
        print("Generating finished.")
