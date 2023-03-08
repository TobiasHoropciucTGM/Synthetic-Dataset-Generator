import argparse
from generators import ClassificationGenerator
from custom_exceptions import InvalidGeneratorArgumentException


parser = argparse.ArgumentParser(description='Augments images for image classifcation to improve dataset or/and it increase if')
parser.add_argument('-s', '--image-source', type=str, help='image directory used as source for augmentations (must only contain image files)', required=True)
parser.add_argument('-d', '--destination', type=str, help='output directory for image augmentations', required=True)
parser.add_argument('-A', '--augmentation', type=str, help='augmentation config file to be applied on dataset generation', required=True)
parser.add_argument('-n', '--number', type=int ,help='generates augmented dataset of given size')
args = parser.parse_args()

try:
    generator = ClassificationGenerator(image_source=args.image_source, destination=args.destination, num=args.number, augment_config=args.augmentation)
    generator.generate()
except InvalidGeneratorArgumentException as e:
    print(e.__class__.__name__+": "+str(e))
