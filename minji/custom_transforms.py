import random

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class Normalize(object):
    """Normalize the color range to [0,1] and convert a color image to grayscale if needed"""

    def __init__(self, color=False):
        self.color = color

    def __call__(self, sample):
        image, key_pts = sample["image"], sample["keypoints"]

        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        # convert image to grayscale
        if not self.color:
            image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # if grayscale, expand to 3 channels
            image_copy = np.stack((image_copy,) * 3, axis=-1)

        # scale color range from [0, 255] to [0, 1]
        image_copy = image_copy / 255.0

        # scale keypoints to be centered around 0 with a range of [-2, 2]
        key_pts_copy = (key_pts_copy - image.shape[0] / 2) / (image.shape[0] / 4)

        return {"image": image_copy, "keypoints": key_pts_copy}

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample["image"], sample["keypoints"]

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h))

        # scale the pts, too
        key_pts = key_pts * [new_w / w, new_h / h]

        return {"image": img, "keypoints": key_pts}

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample["image"], sample["keypoints"]

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top : top + new_h, left : left + new_w]

        key_pts = key_pts - [left, top]

        return {"image": image, "keypoints": key_pts}


class FaceCrop(object):
    """Crop out face using the keypoints as reference"""

    def __call__(self, sample):
        image, key_pts = sample["image"], sample["keypoints"]

        image_copy = np.copy(image)

        h, w = image.shape[:2]

        x_max = 0
        x_min = 10000
        y_max = 0
        y_min = 10000

        # Find the coordinates to keypoints at the far left, far right, top and bottom
        # Also check that no keypoints are outside the image
        for coord in key_pts:
            if coord[0] > x_max:
                if coord[0] >= w:
                    x_max = w
                else:
                    x_max = coord[0]
            if coord[0] < x_min:
                if coord[0] < 0:
                    x_min = 0
                else:
                    x_min = coord[0]
            if coord[1] > y_max:
                if coord[1] >= h:
                    y_max = h
                else:
                    y_max = coord[1]
            if coord[1] < y_min:
                if coord[1] < 0:
                    y_min = 0
                else:
                    y_min = coord[1]

        # Set the left corner keypoint as out crop coordinate
        x = int(x_min)
        y = int(y_min)

        # Get height and width of keypoint area
        new_h = int(y_max - y_min)
        new_w = int(x_max - x_min)

        # Set the smallest side equal to the largest since we want a square
        if new_h > new_w:
            new_w = new_h
        else:
            new_h = new_w

        randsize1 = [2, 70]
        randsize2 = [2, 30]
        randsize3 = [1, 10]

        # Check that padding doesn't go outside the frame
        padding_x_1 = 0
        padding_x_2 = 0
        padding_y_1 = 0
        padding_y_2 = 0

        padding_size_x_1 = random.randint(randsize1[0], randsize1[1])
        padding_size_x_2 = random.randint(randsize1[0], randsize1[1])
        padding_size_y_1 = random.randint(randsize1[0], randsize1[1])
        padding_size_y_2 = random.randint(randsize1[0], randsize1[1])
        if (
            y - padding_size_y_1 > 0
            and x - padding_size_x_1 > 0
            and x + new_w + padding_size_x_2 < w
            and y + new_h + padding_size_y_2 < h
        ):
            padding_x_1 = padding_size_x_1
            padding_x_2 = padding_size_x_2
            padding_y_1 = padding_size_y_1
            padding_y_2 = padding_size_y_2
        else:
            padding_size_x_1 = random.randint(randsize2[0], randsize2[1])
            padding_size_x_2 = random.randint(randsize2[0], randsize2[1])
            padding_size_y_1 = random.randint(randsize2[0], randsize2[1])
            padding_size_y_2 = random.randint(randsize2[0], randsize2[1])

            if (
                y - padding_size_y_1 > 0
                and x - padding_size_x_1 > 0
                and x + new_w + padding_size_x_2 < w
                and y + new_h + padding_size_y_2 < h
            ):
                padding_x_1 = padding_size_x_1
                padding_x_2 = padding_size_x_2
                padding_y_1 = padding_size_y_1
                padding_y_2 = padding_size_y_2

            else:
                padding_size_x_1 = random.randint(randsize3[0], randsize3[1])
                padding_size_x_2 = random.randint(randsize3[0], randsize3[1])
                padding_size_y_1 = random.randint(randsize3[0], randsize3[1])
                padding_size_y_2 = random.randint(randsize3[0], randsize3[1])

                if (
                    y - padding_size_y_1 > 0
                    and x - padding_size_x_1 > 0
                    and x + new_w + padding_size_x_2 < w
                    and y + new_h + padding_size_y_2 < h
                ):
                    padding_x_1 = padding_size_x_1
                    padding_x_2 = padding_size_x_2
                    padding_y_1 = padding_size_y_1
                    padding_y_2 = padding_size_y_2

        image_copy = image_copy[
            y - padding_y_1 : y + new_h + padding_y_2,
            x - padding_x_1 : x + new_w + padding_x_2,
        ]

        key_pts = key_pts - [x - padding_x_1, y - padding_y_1]

        return {"image": image_copy, "keypoints": key_pts}


class FaceCropTight(object):
    """
    Crop out face using the keypoints as reference
    """

    def __call__(self, sample):
        image, key_pts = sample["image"], sample["keypoints"]

        image_copy = np.copy(image)

        h, w = image.shape[:2]

        x_max = 0
        x_min = 10000
        y_max = 0
        y_min = 10000

        # Find the coordinates to keypoints at the far left, far right, top and bottom
        # Also check that no keypoints are outside the image
        for coord in key_pts:
            if coord[0] > x_max:
                if coord[0] >= w:
                    x_max = w
                else:
                    x_max = coord[0]
            if coord[0] < x_min:
                if coord[0] < 0:
                    x_min = 0
                else:
                    x_min = coord[0]
            if coord[1] > y_max:
                if coord[1] >= h:
                    y_max = h
                else:
                    y_max = coord[1]
            if coord[1] < y_min:
                if coord[1] < 0:
                    y_min = 0
                else:
                    y_min = coord[1]

        # Set the left corner keypoint as out crop coordinate
        x = int(x_min)
        y = int(y_min)

        # Get height and width of keypoint area
        new_h = int(y_max - y_min)
        new_w = int(x_max - x_min)

        # Set the smallest side equal to the largest since we want a square
        if new_h > new_w:
            new_w = new_h
        else:
            new_h = new_w

        randsize1 = [5, 10]

        # Check that padding doesn't go outside the frame
        padding_x_1 = 0
        padding_x_2 = 0
        padding_y_1 = 0
        padding_y_2 = 0

        padding_size_x_1 = random.randint(randsize1[0], randsize1[1])
        padding_size_x_2 = random.randint(randsize1[0], randsize1[1])
        padding_size_y_1 = random.randint(randsize1[0], randsize1[1])
        padding_size_y_2 = random.randint(randsize1[0], randsize1[1])

        if (
            y - padding_size_y_1 > 0
            and x - padding_size_x_1 > 0
            and x + new_w + padding_size_x_2 < w
            and y + new_h + padding_size_y_2 < h
        ):
            padding_x_1 = padding_size_x_1
            padding_x_2 = padding_size_x_2
            padding_y_1 = padding_size_y_1
            padding_y_2 = padding_size_y_2

        image_copy = image_copy[
            y - padding_y_1 : y + new_h + padding_y_2,
            x - padding_x_1 : x + new_w + padding_x_2,
        ]

        key_pts = key_pts - [x - padding_x_1, y - padding_y_1]

        return {"image": image_copy, "keypoints": key_pts}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, key_pts = sample["image"], sample["keypoints"]

        # if image has no grayscale color channel, add one
        if len(image.shape) == 2:
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)
            image = np.repeat(image, 3, axis=2)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {
            "image": torch.from_numpy(image),
            "keypoints": torch.from_numpy(key_pts),
        }

class Random90DegFlip(object):
    """Random 90 degree flip of image in sample"""

    def __call__(self, sample):
        image, key_pts = sample["image"], sample["keypoints"]

        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        if random.choice([0, 1]) < 0.25:
            image_copy = np.rot90(image_copy, 1)
            image_copy = np.flipud(image_copy)
            key_pts_copy = np.fliplr(key_pts_copy)

        return {"image": image_copy, "keypoints": key_pts_copy}


class RandomGamma(object):
    """Random gamma of image in sample"""

    def __call__(self, sample):
        image, key_pts = sample["image"], sample["keypoints"]

        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        image_copy = adjust_gamma(image_copy, gamma=random.uniform(0.8, 1.1))

        return {"image": image_copy, "keypoints": key_pts_copy}


class ColorJitter(object):
    """ColorJitter image in sample"""

    def __call__(self, sample):
        image, key_pts = sample["image"], sample["keypoints"]

        color_jitter = transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
        )

        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        image_copy = color_jitter(Image.fromarray(image_copy))
        # image_copy = color_jitter(Image.fromarray((image_copy * 255).astype(np.uint8)))
        image_copy = np.array(image_copy)

        return {"image": image_copy, "keypoints": key_pts_copy}


def adjust_gamma(
    image, gamma=1.0
):  # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

class RandomHorizontalFlip(object):
    """Random horizontal flip of image in sample"""

    def __call__(self, sample):
        image, key_pts = sample["image"], sample["keypoints"]

        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)
        key_pts_copy_2 = np.copy(key_pts_copy)

        if random.choice([0, 1]) <= 0.5:
            # horizontally flip image
            image_copy = np.fliplr(image_copy)
            # keypoints (x,y) = (-x,y)
            key_pts_copy[:, 0] = -key_pts_copy[:, 0]
            # move keypoints form 2 kvadrant to 1 kvadrant
            key_pts_copy[:, 0] = key_pts_copy[:, 0] + image_copy.shape[1]

            # since the keypoints are flipped around the y-axis
            # their placement are wrong int the keypoint array.
            # E.g. the right eye and left eye is in the wrong place,
            # so the keypoints need to be correctly mirrored in the list

            key_pts_copy_2 = np.copy(key_pts_copy)

            # Adjust the key points according to the specific keypoints for cat faces
            # Adjusting the indices for CatFLW dataset
            # For example, assuming indices [0:16] are for left part and [32:48] are for right part
            key_pts_copy_2[:16] = key_pts_copy[32:48][::-1]
            key_pts_copy_2[32:48] = key_pts_copy[:16][::-1]
            key_pts_copy_2[16:32] = key_pts_copy[16:32][::-1]

        return {"image": image_copy, "keypoints": key_pts_copy_2}


# inspired by https://github.com/macbrennan90/facial-keypoint-detection/blob/master/CV_project.ipynb and
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
class Rotate(object):
    """Rotate image in sample by an angle"""

    def __init__(self, rotation):
        self.rotation = rotation

    def __call__(self, sample):
        image, key_pts = sample["image"], sample["keypoints"]

        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        rows = image.shape[0]
        cols = image.shape[1]

        M = cv2.getRotationMatrix2D((rows / 2, cols / 2), self.rotation, 1)
        image_copy = cv2.warpAffine(image_copy, M, (cols, rows))

        key_pts_copy = key_pts_copy.reshape((1, 96))  # 48개의 키포인트 * 2 (x, y 좌표)
        new_keypoints = np.zeros(96)

        for i in range(48):  # 48개의 키포인트
            coord_idx = 2 * i
            old_coord = key_pts_copy[0][coord_idx : coord_idx + 2]
            new_coord = np.matmul(M, np.append(old_coord, 1))
            new_keypoints[coord_idx] += new_coord[0]
            new_keypoints[coord_idx + 1] += new_coord[1]

        new_keypoints = new_keypoints.reshape((48, 2))

        return {"image": image_copy, "keypoints": new_keypoints}


class RandomRotate(object):
    """Rotate image in sample by an angle"""

    def __init__(self, rotation=30):
        self.rotation = rotation

    def __call__(self, sample):
        image, key_pts = sample["image"], sample["keypoints"]

        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        rows = image.shape[0]
        cols = image.shape[1]

        M = cv2.getRotationMatrix2D(
            (rows / 2, cols / 2), random.choice([-self.rotation, self.rotation]), 1
        )
        image_copy = cv2.warpAffine(image_copy, M, (cols, rows))

        key_pts_copy = key_pts_copy.reshape((1, 96))  # 48개의 키포인트 * 2 (x, y 좌표)
        new_keypoints = np.zeros(96)

        for i in range(48):  # 48개의 키포인트
            coord_idx = 2 * i
            old_coord = key_pts_copy[0][coord_idx : coord_idx + 2]
            new_coord = np.matmul(M, np.append(old_coord, 1))
            new_keypoints[coord_idx] += new_coord[0]
            new_keypoints[coord_idx + 1] += new_coord[1]

        new_keypoints = new_keypoints.reshape((48, 2))

        return {"image": image_copy, "keypoints": new_keypoints}


class NormalizeOriginal(object):
    """Convert a color image to grayscale and normalize the color range to [0,1]."""

    def __call__(self, sample):
        image, key_pts = sample["image"], sample["keypoints"]

        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        # convert image to grayscale
        image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # scale color range from [0, 255] to [0, 1]
        image_copy = image_copy / 255.0

        # scale keypoints to be centered around 0 with a range of [-1, 1]
        # mean = 100, sqrt = 50, so, pts should be (pts - 100)/50
        key_pts_copy = (key_pts_copy - 100) / 50.0

        return {"image": image_copy, "keypoints": key_pts_copy}
