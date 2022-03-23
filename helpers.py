"""Source: https://github.com/facebookresearch/detectron2 With minor custom changes"""
import sys

import matplotlib.colors as mplc
import matplotlib as mpl
from detectron2.utils.visualizer import GenericMask
from detectron2.utils.visualizer import ColorMode

p = subprocess.run(shlex.split('pip show detectron2'),
                   shell=False,
                   check=True,
                   capture_output=True,
                   text=True)
utils_path = p.stdout.split('Location')[1].split('\n')[0].split(
    ' ')[1] + 'detectron2/utils'
sys.path.insert(0, utils_path)

from colormap import random_color  # local module

_SMALL_OBJECT_AREA_THRESH = 1000
_LARGE_MASK_AREA_THRESH = 120000
_OFF_WHITE = (1.0, 1.0, 240.0 / 255)
_BLACK = (0, 0, 0)
_RED = (1.0, 0, 0)

_KEYPOINT_THRESHOLD = 0.05


def _create_text_labels(classes, scores, class_names, is_crowd=None):
    """
    Args:
        classes (list[int] or None):
        scores (list[float] or None):
        class_names (list[str] or None):
        is_crowd (list[bool] or None):

    Returns:
        list[str] or None
    """
    labels = None
    if classes is not None:
        if class_names is not None and len(class_names) > 0:
            labels = [class_names[i] for i in classes]
        else:
            labels = [str(i) for i in classes]
    if scores is not None:
        if labels is None:
            labels = ["{:.0f}%".format(s * 100) for s in scores]
        else:
            labels = [
                "{} {:.0f}%".format(l, s * 100)
                for l, s in zip(labels, scores)
            ]
    if labels is not None and is_crowd is not None:
        labels = [
            l + ("|crowd" if crowd else "")
            for l, crowd in zip(labels, is_crowd)
        ]
        print(labels)
    return labels


def draw_polygon(self,
                 segment,
                 color,
                 edge_color=None,
                 alpha=0.5,
                 _linewidth=None):
    """
    Args:
        segment: numpy array of shape Nx2, containing all the points in the polygon.
        color: color of the polygon. Refer to `matplotlib.colors` for a full list of
            formats that are accepted.
        edge_color: color of the polygon edges. Refer to `matplotlib.colors` for a
            full list of formats that are accepted. If not provided, a darker shade
            of the polygon color will be used instead.
        alpha (float): blending efficient. Smaller values lead to more transparent masks.

    Returns:
        output (VisImage): image object with polygon drawn.
    """
    if edge_color is None:
        # make edge color darker than the polygon color
        if alpha > 0.8:
            edge_color = self._change_color_brightness(color,
                                                       brightness_factor=-0.7)
        else:
            edge_color = color
    edge_color = mplc.to_rgb(edge_color) + (1, )

    if not _linewidth:
        _linewidth = max(self._default_font_size // 15 * self.output.scale, 1)
    polygon = mpl.patches.Polygon(
        segment,
        fill=True,
        facecolor=mplc.to_rgb(color) + (alpha, ),
        edgecolor=edge_color,
        linewidth=_linewidth,
    )
    self.output.ax.add_patch(polygon)
    return self.output


def overlay_instances(self,
                      *,
                      boxes=None,
                      labels=None,
                      masks=None,
                      keypoints=None,
                      assigned_colors=None,
                      alpha=0.5,
                      _linewidth=None):
    """
    Args:
        boxes (Boxes, RotatedBoxes or ndarray): either a :class:`Boxes`,
            or an Nx4 numpy array of XYXY_ABS format for the N objects in a single image,
            or a :class:`RotatedBoxes`,
            or an Nx5 numpy array of (x_center, y_center, width, height, angle_degrees) format
            for the N objects in a single image,
        labels (list[str]): the text to be displayed for each instance.
        masks (masks-like object): Supported types are:

            * :class:`detectron2.structures.PolygonMasks`,
                :class:`detectron2.structures.BitMasks`.
            * list[list[ndarray]]: contains the segmentation masks for all objects in one image.
                The first level of the list corresponds to individual instances. The second
                level to all the polygon that compose the instance, and the third level
                to the polygon coordinates. The third level should have the format of
                [x0, y0, x1, y1, ..., xn, yn] (n >= 3).
            * list[ndarray]: each ndarray is a binary mask of shape (H, W).
            * list[dict]: each dict is a COCO-style RLE.
        keypoints (Keypoint or array like): an array-like object of shape (N, K, 3),
            where the N is the number of instances and K is the number of keypoints.
            The last dimension corresponds to (x, y, visibility or score).
        assigned_colors (list[matplotlib.colors]): a list of colors, where each color
            corresponds to each mask or box in the image. Refer to 'matplotlib.colors'
            for full list of formats that the colors are accepted in.

    Returns:
        output (VisImage): image object with visualizations.
    """
    num_instances = 0
    if boxes is not None:
        boxes = self._convert_boxes(boxes)
        num_instances = len(boxes)
    if masks is not None:
        masks = self._convert_masks(masks)
        if num_instances:
            assert len(masks) == num_instances
        else:
            num_instances = len(masks)
    if keypoints is not None:
        if num_instances:
            assert len(keypoints) == num_instances
        else:
            num_instances = len(keypoints)
        keypoints = self._convert_keypoints(keypoints)
    if labels is not None:
        assert len(labels) == num_instances
    if assigned_colors is None:
        assigned_colors = [
            random_color(rgb=True, maximum=1) for _ in range(num_instances)
        ]
    if num_instances == 0:
        return self.output
    if boxes is not None and boxes.shape[1] == 5:
        return self.overlay_rotated_instances(boxes=boxes,
                                              labels=labels,
                                              assigned_colors=assigned_colors)

    # Display in largest to smallest order to reduce occlusion.
    areas = None
    if boxes is not None:
        areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
    elif masks is not None:
        areas = np.asarray([x.area() for x in masks])

    if areas is not None:
        sorted_idxs = np.argsort(-areas).tolist()
        # Re-order overlapped instances in descending order.
        boxes = boxes[sorted_idxs] if boxes is not None else None
        labels = [labels[k]
                  for k in sorted_idxs] if labels is not None else None
        masks = [masks[idx]
                 for idx in sorted_idxs] if masks is not None else None
        assigned_colors = [assigned_colors[idx] for idx in sorted_idxs]
        keypoints = keypoints[sorted_idxs] if keypoints is not None else None

    for i in range(num_instances):
        color = assigned_colors[i]
        if boxes is not None:
            self.draw_box(boxes[i], edge_color=color)

        if masks is not None:
            for segment in masks[i].polygons:
                draw_polygon(v,
                             segment.reshape(-1, 2),
                             color,
                             alpha=alpha,
                             _linewidth=_linewidth)

        if labels is not None:
            # first get a box
            if boxes is not None:
                x0, y0, x1, y1 = boxes[i]
                text_pos = (x0, y0
                            )  # if drawing boxes, put text on the box corner.
                horiz_align = "left"
            elif masks is not None:
                # skip small mask without polygon
                if len(masks[i].polygons) == 0:
                    continue

                x0, y0, x1, y1 = masks[i].bbox()

                # draw text in the center (defined by median) when box is not drawn
                # median is less sensitive to outliers.
                text_pos = np.median(masks[i].mask.nonzero(), axis=1)[::-1]
                horiz_align = "center"
            else:
                continue  # drawing the box confidence for keypoints isn't very useful.
            # for small objects, draw text at the side to avoid occlusion
            instance_area = (y1 - y0) * (x1 - x0)
            if (instance_area < _SMALL_OBJECT_AREA_THRESH * self.output.scale
                    or y1 - y0 < 40 * self.output.scale):
                if y1 >= self.output.height - 5:
                    text_pos = (x1, y0)
                else:
                    text_pos = (x0, y1)

            height_ratio = (y1 - y0) / np.sqrt(
                self.output.height * self.output.width)
            lighter_color = self._change_color_brightness(
                color, brightness_factor=0.7)
            font_size = (np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2) *
                         0.5 * self._default_font_size)
            self.draw_text(
                labels[i],
                text_pos,
                color=lighter_color,
                horizontal_alignment=horiz_align,
                font_size=font_size,
            )

    # draw keypoints
    if keypoints is not None:
        for keypoints_per_instance in keypoints:
            self.draw_and_connect_keypoints(keypoints_per_instance)

    return self.output


def draw_instance_predictions(self,
                              predictions,
                              _alpha,
                              _colors=None,
                              labels=None,
                              _linewidth=None):
    """
    Draw instance-level prediction results on an image.

    Args:
        predictions (Instances): the output of an instance detection/segmentation
            model. Following fields will be used to draw:
            "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").

    Returns:
        output (VisImage): image object with visualizations.
    """
    boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
    scores = predictions.scores if predictions.has("scores") else None
    classes = predictions.pred_classes.tolist() if predictions.has(
        "pred_classes") else None
    if not labels:
        labels = _create_text_labels(classes, scores,
                                     self.metadata.get("thing_classes", None))
    keypoints = predictions.pred_keypoints if predictions.has(
        "pred_keypoints") else None

    if predictions.has("pred_masks"):
        masks = np.asarray(predictions.pred_masks)
        masks = [
            GenericMask(x, self.output.height, self.output.width)
            for x in masks
        ]
    else:
        masks = None

    if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get(
            "thing_colors"):
        colors = [
            self._jitter([x / 255 for x in self.metadata.thing_colors[c]])
            for c in classes
        ]
        alpha = 0.8
    else:
        colors = _colors
        alpha = 0.5

    if self._instance_mode == ColorMode.IMAGE_BW:
        self.output.reset_image(
            self._create_grayscale_image((
                predictions.pred_masks.any(dim=0) > 0
            ).numpy() if predictions.has("pred_masks") else None))
        alpha = 0.3

    overlay_instances(v,
                      masks=masks,
                      boxes=boxes,
                      labels=labels,
                      keypoints=keypoints,
                      assigned_colors=colors,
                      alpha=_alpha,
                      _linewidth=_linewidth)
    return self.output
