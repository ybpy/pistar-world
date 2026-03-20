import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


def make_piper_example() -> dict:
    """Creates a random input example for Piper policy."""
    return {
        "state": np.ones((7,), dtype=np.float32),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class PiperInputs(transforms.DataTransformFn):
    """Inputs for Piper single-arm policy.
    
    Expected inputs:
    - images: dict with "cam_high" and "cam_wrist"
    - state: [7] (6 joints + 1 gripper)
    - actions: [action_horizon, 7]
    """

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_wrist")

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"], dtype=np.float32)

        def convert_image(img):
            img = np.asarray(img)
            if np.issubdtype(img.dtype, np.floating):
                img = (255 * img).astype(np.uint8)
            if len(img.shape) == 3 and img.shape[0] in (1, 3):
                img = einops.rearrange(img, "c h w -> h w c")
            return img

        in_images = data["images"]
        
        base_image = convert_image(in_images["cam_high"])
        
        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        if "cam_wrist" in in_images:
            images["left_wrist_0_rgb"] = convert_image(in_images["cam_wrist"])
            image_masks["left_wrist_0_rgb"] = np.True_
        else:
            images["left_wrist_0_rgb"] = np.zeros_like(base_image)
            image_masks["left_wrist_0_rgb"] = np.False_

        if "cam_wrist1" in in_images:
            images["right_wrist_0_rgb"] = convert_image(in_images["cam_wrist1"])
            image_masks["right_wrist_0_rgb"] = np.True_
        else:
            images["right_wrist_0_rgb"] = np.zeros_like(base_image)
            image_masks["right_wrist_0_rgb"] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        if "adv_ind" in data:
            inputs["adv_ind"] = data["adv_ind"]

        return inputs


@dataclasses.dataclass(frozen=True)
class PiperOutputs(transforms.DataTransformFn):
    """Outputs for Piper policy."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, :7], dtype=np.float32)
        return {"actions": actions}
