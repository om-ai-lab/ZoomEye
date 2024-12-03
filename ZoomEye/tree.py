from PIL import Image
from typing import NamedTuple, List, Optional
import itertools


class ZoomState(NamedTuple):
    original_image_pil: Image.Image
    bbox: List[int]

class Node:
    id_iter = itertools.count()

    @classmethod
    def reset_id(cls):
        cls.id_iter = itertools.count()

    def __init__(
        self, 
        state: Optional[ZoomState], 
        parent: "Optional[Node]" = None, 
        fast_confidence: float = None, 
        fast_confidence_details=None, 
        is_terminal: bool = False
    ) -> None:
        
        self.id = next(Node.id_iter)
        if fast_confidence_details is None:
            fast_confidence_details = {}
        self.confidence_details = {}
        self.cum_confidences: list[float] = []
        self.fast_confidence = self.confidence = fast_confidence
        self.fast_confidence_details = fast_confidence_details
        self.answering_confidence = 0

        self.is_terminal = is_terminal
        self.state = state
        self.parent = parent
        self.children: 'Optional[list[Node]]' = []
        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1

    @property
    def is_leaf(self):
        return len(self.children) == 0

    @property
    def is_root(self):
        return self.depth == 0

    def add_child(self, child: 'Node'):
        self.children.append(child)
    
    def save_crop(self, path):
        x,y,w,h = self.state.bbox
        crop_image = self.state.original_image_pil.crop([x,y,x+w,y+h])
        crop_image.save(path)

def is_terminal(node: Node, smallest_size: int) -> bool:
    now_w, now_h = node.state.bbox[2:]
    return max(now_w, now_h) < smallest_size

class ImageTree:
    def __init__(self, image_path, patch_size):
        image_pil = Image.open(image_path).convert('RGB')
        self.image_pil = image_pil
        self.patch_size = patch_size
        self.root = Node(ZoomState(image_pil, [0, 0, image_pil.width, image_pil.height]))
        self.max_depth = 0
        self._build()
        
    
    def _build(self):
        self._build_recursive(self.root)
    
    def _build_recursive(self, node: Node):
        self.max_depth = max(self.max_depth, node.depth)
        if is_terminal(node, self.patch_size):
            return
        sub_patches, _, _ = get_sub_patches(node.state.bbox, *split_4subpatches(node.state.bbox))
        for sub_patch in sub_patches:   
            next_state = ZoomState(
                original_image_pil=node.state.original_image_pil,
                bbox=sub_patch,
            )
            node.add_child(Node(
                state=next_state,
                parent=node,
            ))

        for child in node.children:
            self._build_recursive(child)



def get_sub_patches(current_patch_bbox, num_of_width_patches, num_of_height_patches):
	width_stride = int(current_patch_bbox[2]//num_of_width_patches)
	height_stride = int(current_patch_bbox[3]/num_of_height_patches)
	sub_patches = []
	for j in range(num_of_height_patches):
		for i in range(num_of_width_patches):
			sub_patch_width = current_patch_bbox[2] - i*width_stride if i == num_of_width_patches-1 else width_stride
			sub_patch_height = current_patch_bbox[3] - j*height_stride if j == num_of_height_patches-1 else height_stride
			sub_patch = [current_patch_bbox[0]+i*width_stride, current_patch_bbox[1]+j*height_stride, sub_patch_width, sub_patch_height]
			sub_patches.append(sub_patch)
	return sub_patches, width_stride, height_stride

def split_4subpatches(current_patch_bbox):
	hw_ratio = current_patch_bbox[3] / current_patch_bbox[2]
	if hw_ratio >= 2:
		return 1, 4
	elif hw_ratio <= 0.5:
		return 4, 1
	else:
		return 2, 2
