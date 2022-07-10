from typing import Dict, List, Optional, Tuple, Union

import einops
import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torchvision.transforms.functional as TF
from torch import Tensor
import math
from torchvision.models import detection
import random
import numpy as np


def sample_frames(total: int, num: int, offset: int = 0) -> List[int]:
    """Uniformly sample num frames from video
    
    Args:
        total: int, total frames, 
        num: int, number of frames to sample
        offset: int, offset from start of video
    Returns: 
        list of frame indices starting from offset
    """

    if total < num:
        # repeat frames if total < num
        repeats = math.ceil(num / total)
        data = [x for x in range(total) for _ in range(repeats)]
        total = len(data)
    else:
        data = list(range(total))
    interval = total // num
    indices = np.arange(0, total, interval)[:num]
    for i, x in enumerate(indices):
        rand = np.random.randint(0, interval)
        if i == num - 1:
            upper = total
            rand = np.random.randint(0, upper - x)
        else:
            upper = min(interval * (i + 1), total)
        indices[i] = (x + rand) % upper
    assert len(indices) == num, f'len(indices)={len(indices)}'
    for i in range(1, len(indices)):
        assert indices[i] > indices[i - 1], f'indices[{i}]={indices[i]}'
    assert num == len(indices), f'num={num}'
    return [data[i] + offset for i in indices]


class Pipeline:
    """Pipeline for data processing and augmentation

    Args:
        crop_size: Tuple[int, int], crop size
        scale_size: Tuple[int, int], scale size
        mean: Tuple[float, float, float], mean of RGB
        std: Tuple[float, float, float], std of RGB
    """

    def __init__(self,
                 scale_size: Tuple[int, int] = (256, 256),
                 crop_size: Tuple[int, int] = (224, 224),
                 mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                 std: Tuple[float, float, float] = (0.229, 0.224, 0.225)):
        self.pipeline_read_video = T.Compose([
            T.ConvertImageDtype(torch.float32),
            T.Resize(scale_size),
            T.RandomCrop(crop_size),
            T.RandomHorizontalFlip(),
            T.Normalize(mean, std),
        ])
        self.pipeline_read_image = T.Compose([
            T.ToPILImage(),
            T.Resize(scale_size),
            T.RandomCrop(crop_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
        self.normalize = T.Normalize(mean, std)

    def transform_read_video(self, frames: Tensor, samples: int = 8) -> Tensor:
        """Transform `torchvision.io.read_video()` output frames
        
        Args:
            frames (Tensor): frame to transform
            samples (int): number of frames to sample from video, if -1, use all frames
        Returns:
            Tensor, transformed frame
        """
        if samples > 0:
            indices = sample_frames(frames.shape[0], samples)
            frames = frames[indices]
        frames = einops.rearrange(frames, 'b h w c -> b c h w')
        return self.pipeline_read_video(frames)

    def transform_read_image(self, image: Tensor) -> Tensor:
        """Transform `torchvision.io.read_image()` output image
        
        Args:
            image (Tensor): images to transform
            
        Returns:
            Tensor, transformed images
        """

        return self.pipeline_read_image(image)


class Detector:
    """Human detector. Generates bbox of persons in images and videos.
    
    Args:
        model_name (str): model name in torchvision.models.detection
        device (str): cpu or cuda
   
    Example::

        >>> det = Detector()
        >>> img = Image.open('test.jpg')
        >>> box_tensor = det.detect(TF.convert_image_dtype(img, torch.float32))
        >>> box_tensor.shape
        torch.Size([1, 1, 4])
        >>> box_tensor
        tensor([[[ 119.2578,  478.0018,  365.2753, 1199.4677]]], device='cuda:0')
    """

    def __init__(self, model_name: str = 'fasterrcnn_resnet50_fpn', device: str = 'cuda'):
        self.model = getattr(detection, model_name)(pretrained=True)
        self.model.to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def detect(self,
               images: Union[Tensor, List[Tensor]],
               threshold: float = 0.7) -> List[Tensor]:
        """Detect human. Returns a list of bboxs of input frames.

        Args:
            images (List[Tensor]): input images of type torch.float in range [0, 1].
            threshold (float): threshold. Defaults to 0.7.
        Returns:
            Tensor: stacked list of bounding boxes of shape (num_frames, num_person, 4)
        """
        if isinstance(images, Tensor):
            images = images.to(self.device)
            if images.dim() == 3:
                images = [images]
        assert images[0][..., -2:].dtype == torch.float, 'images must be float'
        results: List[Dict[str, Tensor]] = self.model(images)
        persons: List[Tensor] = []
        for r in results:
            persons.append(self._get_box_one_frame(r, threshold))
        return persons

    def _get_box_one_frame(self, result: dict, threshold: float = 0.7) -> Tensor:
        """Get bounding boxes of one frame.
        Returns zeros if no person is detected.
        
        Args:
            result (dict): detection result of one frame
            threshold (float): threshold. Defaults to 0.7.
        Returns:
            Tensor: bounding boxes of shape (N, 4)
        """

        human_inds = result['labels'] == 1
        person_boxes = result['boxes'][human_inds]
        scores = result['scores'][human_inds]
        person_boxes = person_boxes[scores > threshold]
        if person_boxes.shape[0] == 0:
            return torch.zeros((1, 4))
        return person_boxes

    def crop_person(self, images: List[Tensor]) -> List[Tensor]:
        """Crop one person from images.

        Args:
            image, List[Tensor]: image to crop person from.
        Returns:
            List[Tensor]: cropped image padded or resized to 224x224.
        Note:
            For now, simply take the largest bbox from images. Ignores temporal information.
        TODO:
            Person tracking in video
        """

        boxes = self.detect(images)
        cropped_images: List[Tensor] = []
        for persons in boxes:
            x1, y1, x2, y2 = persons[0]
            # make box larger by 10%
            x1, y1, x2, y2 = list(map(int, [x1 * 0.9, y1 * 0.9, x2 * 1.1, y2 * 1.1]))
            person = TF.crop(images[0], x1, y1, x2 - x1, y2 - y1)
            h, w = person.shape[:2]
            max_hw = max(h, w)
            person = TF.pad(person,
                            padding=(max_hw - h, max_hw - w),
                            fill=0,
                            padding_mode='constant')
            person = TF.resize(person, size=(224, 224))
            cropped_images.append(person)
            assert person.shape[-2:] == (224, 224), f"{person.shape}"
        return cropped_images


class PersonCrop:
    """Crop one person from images. Can't asure the same person though.
    If no person detected, return original image.
    Person bboxes are enlarged by 10%.
    For one group of images, x, y, w, h are fixed to the largest bbox that can 
        cover all first detected bboxes.

    Returns:
        Tensor: shape (..., box_h, box_w)

    Algorithm:
        #TODO:
            Track person by calculating centroid or IoU.
            If bboxes too crowded, return original image.
    """

    def __init__(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.worker = Detector('fasterrcnn_resnet50_fpn', 'cpu')

    def __call__(self, images: Tensor) -> Tensor:
        box_tensor = torch.stack([b[0] for b in self.worker.detect(images)])  # First box
        x1, y1 = box_tensor[:, 0].min().item(), box_tensor[:, 1].min().item()
        x2, y2 = box_tensor[:, 2].max().item(), box_tensor[:, 3].max().item()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        if w * h == 0:
            return images
        # Enlarge bbox by 10%
        x1, y1 = int(x1 - w * 0.05), int(y1 - h * 0.05)
        w, h = int(w * 1.1), int(h * 1.1)
        return TF.crop(images, y1, x1, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ThreeCrop:
    """Crop three square images along the larger side with equal intervals.
    And randomly returns one of the three images.
    If width > height, crop three horizontally. Otherwise, crop three vertically.
    The cropped size equals to the shorter side.

    Returns:
        Tensor: shape (..., H, W)
    """

    def __init__(self):
        pass

    def __call__(self, images: Tensor) -> Tensor:
        h, w = images.shape[-2:]
        th = tw = min(h, w)
        if h > w:
            step = (h - w) // 2
            top_left = [(0, 0), (step, 0), (2 * step, 0)]
        else:
            step = (w - h) // 2
            top_left = [(0, 0), (0, step), (0, 2 * step)]
        sample = random.choice(top_left)
        return TF.crop(images, sample[0], sample[1], th, tw)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class MultiScaleCrop:
    """Crop images with a list of randomly selected scales.

    Generates a list of min(w, h) * scales. Then randomly select one 
    pair of w, h from it.

    Args:
        scales (tuple[float]): width and height scales to be selected.
        max_wh_scale_gap (int): Maximum gap of w and h scale levels.
            Default: 1.
        random_crop (bool): If set to True, the cropping bbox will be randomly
            sampled, otherwise it will be sampler from fixed regions.
            Default: False.
        num_fixed_crops (int): If set to 5, the cropping bbox will keep 5
            basic fixed regions: "upper left", "upper right", "lower left",
            "lower right", "center". If set to 13, the cropping bbox will
            append another 8 fix regions: "center left", "center right",
            "lower center", "upper center", "upper left quarter",
            "upper right quarter", "lower left quarter", "lower right quarter".
            Default: 5.
    """

    def __init__(self,
                 scales: Tuple[float, ...] = (1,),
                 max_wh_scale_gap: int = 1,
                 random_crop: bool = False,
                 num_fixed_crops: int = 5):
        if not isinstance(scales, tuple):
            raise TypeError(f'Scales must be tuple, but got {type(scales)}')

        if num_fixed_crops not in [5, 13]:
            raise ValueError(f'Num_fix_crops must be in {[5, 13]}, '
                             f'but got {num_fixed_crops}')

        self.scales = scales
        self.max_wh_scale_gap = max_wh_scale_gap
        self.random_crop = random_crop
        self.num_fixed_crops = num_fixed_crops

    def __call__(self, images: Tensor) -> Tensor:
        img_h, img_w = images.shape[-2:]
        base_size = min(img_h, img_w)
        crop_sizes = [int(base_size * s) for s in self.scales]

        candidate_sizes = []
        for i, h in enumerate(crop_sizes):
            for j, w in enumerate(crop_sizes):
                if abs(i - j) <= self.max_wh_scale_gap:
                    candidate_sizes.append([w, h])

        crop_size = random.choice(candidate_sizes)
        crop_w, crop_h = crop_size

        if self.random_crop:
            x_offset = random.randint(0, img_w - crop_w)
            y_offset = random.randint(0, img_h - crop_h)
        else:
            w_step = (img_w - crop_w) // 4
            h_step = (img_h - crop_h) // 4
            candidate_offsets = [
                (0, 0),  # upper left
                (4 * w_step, 0),  # upper right
                (0, 4 * h_step),  # lower left
                (4 * w_step, 4 * h_step),  # lower right
                (2 * w_step, 2 * h_step),  # center
            ]
            if self.num_fixed_crops == 13:
                extra_candidate_offsets = [
                    (0, 2 * h_step),  # center left
                    (4 * w_step, 2 * h_step),  # center right
                    (2 * w_step, 4 * h_step),  # lower center
                    (2 * w_step, 0 * h_step),  # upper center
                    (1 * w_step, 1 * h_step),  # upper left quarter
                    (3 * w_step, 1 * h_step),  # upper right quarter
                    (1 * w_step, 3 * h_step),  # lower left quarter
                    (3 * w_step, 3 * h_step)  # lower right quarter
                ]
                candidate_offsets.extend(extra_candidate_offsets)
            x_offset, y_offset = random.choice(candidate_offsets)

        return TF.crop(images, y_offset, x_offset, crop_h, crop_w)

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'input_size={self.input_size}, scales={self.scales}, '
                    f'max_wh_scale_gap={self.max_wh_scale_gap}, '
                    f'random_crop={self.random_crop}, '
                    f'num_fixed_crops={self.num_fixed_crops}')
        return repr_str
