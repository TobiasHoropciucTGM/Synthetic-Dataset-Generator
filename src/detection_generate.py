import argparse
from generators import DetectionGenerator
from custom_exceptions import InvalidGeneratorArgumentException


parser = argparse.ArgumentParser(
    description='Augments images for image classifcation to improve dataset or/and increase datas')
parser.add_argument('-s', '--image-source', type=str,
                    help='images to be used as background for objects (directory must only contain images)',
                    required=True)
parser.add_argument('-o', '--objects-source', type=str,
                    help='detection relevant object images (directory must only contain png images)', required=True)
parser.add_argument('-d', '--destination', type=str, help='output directory for generated images', required=True)
parser.add_argument('-f', '--annotation-format', type=str, help='annotation format to be used (e.g. xml or json)')
parser.add_argument('-a', '--annotation-output', type=str, help='output directory for image annotations', required=True)
parser.add_argument('-n', '--number', type=int, help='desired dataset size to be generated', required=True)
parser.add_argument('-A', '--augmentation', type=str, help='augmentation config file to be applied on dataset generation', required=True)
args = parser.parse_args()
try:
    generator = DetectionGenerator(image_source=args.image_source, object_source=args.objects_source, num=args.number,
                                   destination=args.destination, annotation_format=args.annotation_format , annotation_output_dir=args.annotation_output, augment_config=args.augmentation)
    generator.generate()
except InvalidGeneratorArgumentException as e:
    print(e.__class__.__name__+": "+str(e))