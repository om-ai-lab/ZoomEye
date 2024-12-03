import re
from PIL import Image, ImageDraw
import numpy as np
import spacy
nlp = spacy.load("en_core_web_sm")



def extract_targets(sentence: str, pattern = r"So I need the information about the following objects: (.+)"):
    match = re.search(pattern, sentence)
    if match:
        return match.group(1)
    return None

def split_targets_sentence(targets_sentence:str, split_tag = r' and |, '):
    if targets_sentence.endswith('.'):
        targets_sentence = targets_sentence[:-1]
    targets = re.split(split_tag, targets_sentence)
    return targets

from copy import deepcopy
def expand2square(pil_img, background_color):
	width, height = pil_img.size
	if width == height:
		return deepcopy(pil_img), 0, 0
	elif width > height:
		result = Image.new(pil_img.mode, (width, width), background_color)
		result.paste(pil_img, (0, (width - height) // 2))
		return result, 0, (width - height) // 2
	else:
		result = Image.new(pil_img.mode, (height, height), background_color)
		result.paste(pil_img, ((height - width) // 2, 0))
		return result, (height - width) // 2, 0

def bbox_area(bbox):
    x_min, y_min, x_max, y_max = bbox
    return (x_max - x_min) * (y_max - y_min)

def intersect_bbox(bboxA, bboxB, distance_buffer=50):
    bbox1 = [v-distance_buffer if i<2 else v+distance_buffer for i, v in enumerate(bboxA)]
    bbox2 = [v-distance_buffer if i<2 else v+distance_buffer for i, v in enumerate(bboxB)]
    """ Calculate the union of two bounding boxes. """
    x_min = max(bbox1[0], bbox2[0])
    y_min = max(bbox1[1], bbox2[1])
    x_max = min(bbox1[2], bbox2[2])
    y_max = min(bbox1[3], bbox2[3])

    if x_max > x_min and y_max > y_min:
        return (x_min, y_min, x_max, y_max)
    
    return None

def merge_bboxes(bbox1, bbox2):
    return (
        min(bbox1[0], bbox2[0]),
        min(bbox1[1], bbox2[1]),
        max(bbox1[2], bbox2[2]),
        max(bbox1[3], bbox2[3])
    )


def merge_bbox_list(bboxes, threshold=0):
    """merge all cross bboxes in the List bboxes"""
    changed = True
    while changed:
        changed = False
        new_bboxes = []
        used = set()

        for i in range(len(bboxes)):
            if i in used:
                continue
            merged = False

            for j in range(len(bboxes)):
                if j in used or i == j:
                    continue
                intersection = intersect_bbox(bboxes[i], bboxes[j])
                if intersection:
                    if threshold == 0 or (threshold > 0 and (bbox_area(intersection) >= threshold * bbox_area(bboxes[i]) or bbox_area(intersection) >= threshold * bbox_area(bboxes[j]))):
                        new_bbox = merge_bboxes(bboxes[i], bboxes[j])
                        new_bboxes.append(new_bbox)
                        used.update([i, j])
                        changed = True
                        merged = True
                        break
            if not merged and i not in used:
                new_bboxes.append(bboxes[i])

        bboxes = new_bboxes

    return bboxes

def union_all_bboxes(bboxes):
    if len(bboxes) == 0:
        return None
    ret = bboxes[0]
    for bbox in bboxes[1:]:
        ret = merge_bboxes(ret, bbox)
    return ret


def union_blocks_independent(full_image: Image.Image, bbox1, bbox2, resized_long, backgroud_color):
    if bbox1[0] > bbox2[0]:
        bbox1, bbox2 = bbox2, bbox1
    block1 = full_image.crop(bbox1).resize((resized_long, resized_long))
    block2 = full_image.crop(bbox2).resize((resized_long, resized_long))
    background = Image.new('RGB', (2 * resized_long, 2 * resized_long), backgroud_color)
    center_y1 = (bbox1[3]+bbox1[1]) // 2
    center_y2 = (bbox2[3]+bbox2[1]) // 2
    offset_y = center_y2 - center_y1
    offset_y = np.clip(offset_y, -resized_long, resized_long)
    paste_x1 = 0
    paste_y1 = resized_long//2 - offset_y//2
    paste_x2 = resized_long
    paste_y2 = resized_long//2 + offset_y//2

    background.paste(block1, (paste_x1, paste_y1))
    background.paste(block2, (paste_x2, paste_y2))

    return background


def visualize_bbox_and_arrow(image: Image.Image, bbox, color="red", thickness=2, xyxy=False):
    """Visualizes a single bounding box on the image"""
    if not xyxy:
        x1, y1, w, h = bbox
        x2 = x1 + w
        y2 = y1 + h
    else:
        x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - thickness)
    y1 = max(0, y1 - thickness)
    x2 = min(image.width, x2 + thickness)
    y2 = min(image.height, y2 + thickness)
    draw = ImageDraw.Draw(image)
    new_bbox = [x1, y1, x2, y2]
    draw.rectangle((x1, y1, x2, y2), outline=color, width=thickness)
    min_distance = thickness * 6
    center_x = image.width//2
    center_y = image.height//2
    center_x_bbox = (x1+x2)//2
    center_y_bbox = (y1+y2)//2
    return new_bbox


# For the visual cues like "man and his bag", we should remove the pronoun "his bag"
def include_pronouns(text):
    doc = nlp(text)
    for token in doc:
        if token.pos_ == 'PRON':
            return True
    return False