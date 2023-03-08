import os
from pascal_voc_writer import Writer
import json

class DetectionAnnotations:
    ACCEPTED_ANNOTATION_FORMATS = ['xml', 'json']

    @staticmethod
    def write_xml(image_path, image_width, image_height, labels, boxes, annotation_dir):
        writer = Writer(image_path, image_width, image_height)
        for label, box in zip(labels, boxes):
            writer.addObject(str(label), box['x0'], box['y0'], box['x1'], box['y1'])
        image_name = os.path.basename(image_path)
        writer.save(annotation_dir + image_name[0:image_name.index(".")] + ".xml")

    @staticmethod
    def write_json(image_path, image_width, image_height, labels, boxes, annotation_dir):
        image_name = os.path.basename(image_path)
        annotation_dict = {
            'image_name': image_name,
            'image_width': image_width,
            'image_height': image_height,
            'labels': labels,
            'boxes': boxes
        }
        json_object = json.dumps(annotation_dict, indent=4)
        with open(annotation_dir + image_name[0:image_name.index(".")] + ".json", "w") as outfile:
            outfile.write(json_object)


    @staticmethod
    def accepted_format(annotation_format: str) -> bool:
        return annotation_format.lower() in DetectionAnnotations.ACCEPTED_ANNOTATION_FORMATS

    @staticmethod
    def write_annotations(annotation_format: str, image_path: str, image_width: int, image_height: int, labels: str,
                          boxes: dict, annotation_dir: str):
        match (annotation_format.lower()):
            case 'xml':
                DetectionAnnotations.write_xml(image_path, image_width, image_height, labels, boxes, annotation_dir)
                return True
            case 'json':
                DetectionAnnotations.write_json(image_path, image_width, image_height, labels, boxes, annotation_dir)
                return True