from ZoomEye.zoom_model import ZoomModel, ZoomModelGlobalLocal, BOX_COLOR
from ZoomEye.tree import ImageTree, Node, ZoomState
from ZoomEye.utils import include_pronouns
from typing import Union, Callable, List, Tuple
from PIL import Image
from copy import deepcopy
import os

def get_zoom_eye_response(
        zoom_model: ZoomModel,
        annotation,
        ic_examples,
        decomposed_question_template,
        answering_confidence_threshold_upper,
        answering_confidence_threshold_lower,
        visual_cue_threshold,
        pop_limit,
        threshold_descrease,
        image_folder: str = None,
        mode="eval"
    ):
    input_image = annotation['input_image']
    if image_folder is not None:
        input_image = os.path.join(image_folder, input_image)
    question = annotation['question']
    options = annotation.get('options', None)

    targets = zoom_model.generate_visual_cues_using_ic(ic_examples, question)
    targets = [t for t in targets if not include_pronouns(t)]
    
    searched_nodes = []
    image_pil = Image.open(input_image).convert('RGB')
    image_tree = ImageTree(
        image_path=input_image,
        patch_size=zoom_model.input_size[0],
    )
    zoom_global_local =  isinstance(zoom_model, ZoomModelGlobalLocal)
    if zoom_global_local:
        new_targets = zoom_model.filter_visual_cues(image_pil, targets, image_tree.root, decomposed_question_template, answering_confidence_threshold_upper)
        if len(new_targets) != len(targets):
            # Use the global image to answer
            targets = []
        BOX_COLOR = 'red'
        # Some questions in HR-Bench involve a “red rectangle,” which conflicts with the visual prompt used in the zoomed view. In such cases, we use blue instead.
        if 'red rectangle' in question:
            BOX_COLOR = 'blue'

    if mode == "debug":
        print("targets:", targets)
    one_target_search = (len(targets) == 1 and not targets[0].startswith('all '))
    annotation['targets'] = targets
    annotation['num_pop'] = []
    

    for target in targets:
        smallest_size = zoom_model.input_size[0]
        image_tree = ImageTree(
            image_path=input_image,
            patch_size=smallest_size,
        )
        is_type2 = target.startswith("all ")
        if not is_type2:
            candidates, num_pop = zoom_eye_search_type1(
                zoom_model=zoom_model,
                pop_limit=pop_limit,
                num_intervel=2,
                threshold_descrease=threshold_descrease,
                depth_limit=5,
                question=question if one_target_search else decomposed_question_template.format(target),
                visual_cue=target,
                answering_confidence_threshold_lower=answering_confidence_threshold_lower,
                answering_confidence_threshold_upper=answering_confidence_threshold_upper,
                image_pil=image_pil,
                image_tree=image_tree,
                force_no_root=False if zoom_global_local else (False if one_target_search else True),
                force_visit_bottom=False if one_target_search else True,
            )
            if len(candidates) > 0:
                candidates.sort(key=lambda x: x.depth, reverse=True)
                searched_nodes.append(candidates[0])
                if mode == "debug":
                    zoom_model.save_crop(image_pil, candidates[0], f"demo/{target}.png")
                    print("confidence:", candidates[0].answering_confidence)
                    print("num_pop:", num_pop)
        else:
            target = target[4:]
            # This operation aims to remove plural forms. Simply dropping the “s” is a crude approach; instead, you can use spaCy to lemmatize plural words back to their singular form.
            if target.endswith('s'):
                target = target[:-1]
            candidates = zoom_eye_search_type2(
                depth_limit=2,
                image_tree=image_tree,
                visual_cue=target,
                zoom_model=zoom_model,
                visual_cue_threshold=visual_cue_threshold,
                image_pil=image_pil,
            )
            if len(candidates) > 0:
                searched_nodes.extend(candidates)
                if mode == "debug":
                    for i, candidate in enumerate(candidates):
                        zoom_model.save_crop(image_pil, candidate, f"demo/{target}_i.png")
                        print("confidence:", candidate.fast_confidence)

    annotation['searched_bbox'] = [node.state.bbox for node in searched_nodes]
    answer_type = annotation.get('answer_type', 'free_form')
    # For vstar
    if answer_type == "logits_match":
        option_choose = zoom_model.multiple_choices_inference(image_pil, question, options, searched_nodes)
        return option_choose
    elif answer_type == "free_form":
        response = zoom_model.free_form_using_nodes(image_pil, question, searched_nodes, return_zoomed_view=True)
        return response
    # For hr-bench
    elif answer_type == "option_list":
        answers = []
        for option_str in options:
            question_input = format_question(question, option_str)
            answers.append(zoom_model.free_form_using_nodes(image_pil, question_input, searched_nodes))
        return answers
    # For mme-realworld
    elif answer_type == "Multiple Choice":
        question_input = format_question_multichoice(question, options)
        response = zoom_model.free_form_using_nodes(image_pil, question_input, searched_nodes)
        return response
    else:
        raise NotImplementedError

def format_question(question, option_str):
    return question + '\n' + option_str + 'Answer the option letter directly.'

def format_question_multichoice(question, options):
    ret = question
    for o in options:
        ret += '\n'
        ret += o
    # This prompt is copied from the original paper of MME-RealWorld
    ret += '\nSelect the best answer to the above multiple-choice question based on the image. Respond with only the letter (A, B, C, D, or E) of the correct option.\nThe best answer is:'
    return ret

def zoom_eye_search_type1(
        zoom_model: ZoomModel,
        pop_limit: Union[int, Callable],
        num_intervel: int,
        threshold_descrease: List[float],
        depth_limit: int,
        question: str,
        visual_cue: str,
        answering_confidence_threshold_lower: float,
        answering_confidence_threshold_upper: float,
        force_visit_bottom: bool=False,
        force_no_root: bool=False,
        image_pil: Image.Image = None,
        image_tree: ImageTree = None,
    ):
    max_depth = min(image_tree.max_depth, depth_limit)
    # print("max_depth:", max_depth)
    pop_num_limit = pop_limit(max_depth) if callable(pop_limit) else pop_limit
    Q = []
    root_node = image_tree.root
    root_node.answering_confidence = -1
    Q.append(root_node)
    pop_trace = []
    num_pop = 0
    visit_bottom = False
    candidates = []
    temp_threshold_descrease = deepcopy(threshold_descrease)
    def stopping_criterion(cur_node) -> bool:
        if (not force_no_root) or (force_no_root and not cur_node.is_root):
            cur_answering_confidnce = zoom_model.get_confidence_value(cur_node, image_pil, confidence_type='answering', input_ele=question)
            cur_node.answering_confidence = cur_answering_confidnce
            return cur_answering_confidnce >= answering_confidence_threshold_upper
        return False
    
    def get_priority(node: Node):
        if node.fast_confidence is None:
            existence_confidence = zoom_model.get_confidence_value(node, image_pil, confidence_type='existence', input_ele=visual_cue)
            latent_confidence = zoom_model.get_confidence_value(node, image_pil, confidence_type='latent', input_ele=visual_cue)
            w = zoom_model.get_confidence_weight(node, max_depth)
            node.fast_confidence = existence_confidence * w + latent_confidence * (1 - w)
            node.fast_confidence_details = {
                'existence': existence_confidence,
                'latent': latent_confidence,
                'weight': w
            }
        return node.fast_confidence
    
    def update_candidates(
        new_answering_threshold: float,
        pop_trace: List[Tuple[int, Node]],
        candidates: List[Node],
        num_candidates = 1,
    ):
        for _, node in pop_trace:
            if node.answering_confidence >= new_answering_threshold and node not in candidates:
                candidates.append(node)
        if len(candidates) > num_candidates:
            candidates.sort(key=lambda x: x.answering_confidence, reverse=True)
            while len(candidates) > num_candidates:
                candidates.pop()

    while len(Q) > 0:
        cur_node = Q.pop(0)
        cur_node: Node
        num_pop += 1
        pop_trace.append((0, cur_node))

        if stopping_criterion(cur_node):
            candidates.append(cur_node)
            if not force_visit_bottom:
                break
            if force_visit_bottom and visit_bottom:
                break
        else:
            if num_pop >= pop_num_limit:
                answering_confidence_threshold_upper -= temp_threshold_descrease[0]
                if len(temp_threshold_descrease) > 1:
                    _ = temp_threshold_descrease.pop(0)
                pop_num_limit += num_intervel
                update_candidates(answering_confidence_threshold_upper, pop_trace, candidates)
                if len(candidates) > 0:
                    break
                if answering_confidence_threshold_upper < answering_confidence_threshold_lower :
                    break
        if cur_node.is_leaf or cur_node.depth == max_depth:
            visit_bottom = True
            if force_visit_bottom and len(candidates) > 0:
                break
        else: 
            for child in cur_node.children:
                Q.append(child)
        # Ranking function is implemented here using key=...
        Q.sort(key=lambda x: get_priority(x), reverse=True)
    return candidates, num_pop


def zoom_eye_search_type2(
    depth_limit=2,
    image_tree: ImageTree = None,
    visual_cue: str = None, 
    zoom_model: ZoomModel = None,
    visual_cue_threshold: float = None,
    image_pil: Image = None,
):        
    candidates = []
    _ = post_order_search(image_tree.root, visual_cue, zoom_model, visual_cue_threshold, candidates, image_pil, depth_limit)
    return candidates

INIT_MAX_CONF = -1
def post_order_search(
        node: Node,
        visual_cue: str,
        zoom_model: ZoomModel,
        visual_cue_threshold: float,
        candidates: List[Node],
        image_pil: Image.Image,
        depth_limit: int
    ) -> float:
    max_conf = INIT_MAX_CONF
    for child in node.children:
        if node.depth < depth_limit:
            max_conf = max(max_conf, post_order_search(child, visual_cue, zoom_model, visual_cue_threshold,  candidates, image_pil, depth_limit))
    existence_confidence = zoom_model.get_confidence_value(node, image_pil, confidence_type='existence', input_ele=visual_cue)
    node.fast_confidence = existence_confidence

    if node.fast_confidence >= max(max_conf, visual_cue_threshold):
        candidates.append(node)
    return max(max_conf, node.fast_confidence)


def get_direct_response(
        zoom_model: ZoomModel,
        annotation,
        image_folder
    ):
    input_image = annotation['input_image']
    if image_folder is not None:
        input_image = os.path.join(image_folder, input_image)
    question = annotation['question']
    options = annotation.get('options', None)

    image_pil = Image.open(input_image).convert('RGB')

    # An empty list will conduct direct answering.
    searched_nodes = []
    answer_type = annotation.get('answer_type', 'free_form')
    # For vstar
    if answer_type == "logits_match":
        option_choose = zoom_model.multiple_choices_inference(image_pil, question, options, searched_nodes)
        return option_choose
    elif answer_type == "free_form":
        return zoom_model.free_form_using_nodes(image_pil, question, searched_nodes)
    # For hr-bench
    elif answer_type == "option_list":
        answers = []
        for option_str in options:
            question_input = format_question(question, option_str)
            answers.append(zoom_model.free_form_using_nodes(image_pil, question_input, searched_nodes))
        return answers
    else:
        raise NotImplementedError
