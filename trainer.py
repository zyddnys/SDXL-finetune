# Install bitsandbytes:
# `nvcc --version` to get CUDA version.
# `pip install -i https://test.pypi.org/simple/ bitsandbytes-cudaXXX` to install for current CUDA.
# Example Usage:
# Single GPU: torchrun --nproc_per_node=1 trainer/diffusers_trainer.py --model="CompVis/stable-diffusion-v1-4" --run_name="liminal" --dataset="liminal-dataset" --hf_token="hf_blablabla" --bucket_side_min=64 --use_8bit_adam=True --gradient_checkpointing=True --batch_size=1 --fp16=True --image_log_steps=250 --epochs=20 --resolution=768 --use_ema=True
# Multiple GPUs: torchrun --nproc_per_node=N trainer/diffusers_trainer.py --model="CompVis/stable-diffusion-v1-4" --run_name="liminal" --dataset="liminal-dataset" --hf_token="hf_blablabla" --bucket_side_min=64 --use_8bit_adam=True --gradient_checkpointing=True --batch_size=10 --fp16=True --image_log_steps=250 --epochs=20 --resolution=768 --use_ema=True

import argparse
import copy
import socket
import torch
import torchvision
import transformers
import diffusers
import os
import glob
import random
import tqdm
import resource
import psutil
import pynvml
import wandb
import gc
import time
import itertools
import numpy as np
import PIL
import json
import re
import traceback
import gc
import shutil
import torch.nn.functional as F

from torchvision import datasets, transforms
from torchvision.transforms import functional as visionF

try:
    pynvml.nvmlInit()
except pynvml.nvml.NVMLError_LibraryNotFound:
    pynvml = None

from typing import Iterable, Optional
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, PNDMScheduler, DDIMScheduler, StableDiffusionXLPipeline, EulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
from PIL import Image, ImageOps
from PIL.Image import Image as Img

from collections import defaultdict
from typing import Dict, List, Generator, Tuple
from scipy.interpolate import interp1d

torch.backends.cuda.matmul.allow_tf32 = True

# defaults should be good for everyone
# TODO: add custom VAE support. should be simple with diffusers
bool_t = lambda x: x.lower() in ['true', 'yes', '1']
parser = argparse.ArgumentParser(description='Stable Diffusion Finetuner')
parser.add_argument('--model', type=str, default=None, required=True, help='The name of the model to use for finetuning. Could be HuggingFace ID or a directory')
parser.add_argument('--resume', type=str, default=None, help='The path to the checkpoint to resume from. If not specified, will create a new run.')
parser.add_argument('--run_name', type=str, default=None, required=True, help='Name of the finetune run.')
parser.add_argument('--dataset', type=str, default=None, required=True, help='The path to the dataset to use for finetuning.')
parser.add_argument('--num_buckets', type=int, default=20, help='The number of buckets.')
parser.add_argument('--bucket_side_min', type=int, default=256, help='The minimum side length of a bucket.')
parser.add_argument('--bucket_side_max', type=int, default=1536, help='The maximum side length of a bucket.')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--gradient_accumulation', type=int, default=2, help='gradient_accumulation size default 2')
parser.add_argument('--use_ema', type=bool_t, default='False', help='Use EMA for finetuning')
parser.add_argument('--ucg', type=float, default=0.1, help='Percentage chance of dropping out the text condition per batch. Ranges from 0.0 to 1.0 where 1.0 means 100% text condition dropout.') # 10% dropout probability
parser.add_argument('--gradient_checkpointing', dest='gradient_checkpointing', type=bool_t, default='False', help='Enable gradient checkpointing')
parser.add_argument('--use_8bit_adam', dest='use_8bit_adam', type=bool_t, default='False', help='Use 8-bit Adam optimizer')
parser.add_argument('--adam_beta1', type=float, default=0.9, help='Adam beta1')
parser.add_argument('--adam_beta2', type=float, default=0.999, help='Adam beta2')
parser.add_argument('--adam_weight_decay', type=float, default=0, help='Adam weight decay')
parser.add_argument('--adam_epsilon', type=float, default=1e-08, help='Adam epsilon')
parser.add_argument('--lr_scheduler', type=str, default='cosine', help='Learning rate scheduler [`cosine`, `linear`, `constant`]')
parser.add_argument('--lr_scheduler_warmup', type=float, default=0.001, help='Learning rate scheduler warmup steps. This is a percentage of the total number of steps in the training run. 0.1 means 10 percent of the total number of steps.')
parser.add_argument('--seed', type=int, default=42, help='Seed for random number generator, this is to be used for reproduceability purposes.')
parser.add_argument('--output_path', type=str, default='./output', help='Root path for all outputs.')
parser.add_argument('--save_steps', type=int, default=500, help='Number of steps to save checkpoints at.')
parser.add_argument('--resolution', type=int, default=512, help='Image resolution to train against. Lower res images will be scaled up to this resolution and higher res images will be scaled down.')
parser.add_argument('--shuffle', dest='shuffle', type=bool_t, default='True', help='Shuffle dataset')
parser.add_argument('--hf_token', type=str, default=None, required=False, help='A HuggingFace token is needed to download private models for training.')
parser.add_argument('--project_id', type=str, default='diffusers', help='Project ID for reporting to WandB')
parser.add_argument('--fp16', dest='fp16', type=bool_t, default='False', help='Train in mixed precision')
parser.add_argument('--image_log_steps', type=int, default=500, help='Number of steps to log images at.')
parser.add_argument('--image_log_amount', type=int, default=4, help='Number of images to log every image_log_steps')
parser.add_argument('--image_log_inference_steps', type=int, default=50, help='Number of inference steps to use to log images.')
parser.add_argument('--image_log_scheduler', type=str, default="EulerDiscreteScheduler", help='Number of inference steps to use to log images.')
parser.add_argument('--clip_penultimate', type=bool_t, default='True', help='Use penultimate CLIP layer for text embedding')
parser.add_argument('--output_bucket_info', type=bool_t, default='False', help='Outputs bucket information and exits')
parser.add_argument('--resize', type=bool_t, default='False', help="Resizes dataset's images to the appropriate bucket dimensions.")
parser.add_argument('--use_xformers', type=bool_t, default='False', help='Use memory efficient attention')
parser.add_argument('--wandb', dest='enablewandb', type=bool_t, default='False', help='Enable WeightsAndBiases Reporting')
parser.add_argument('--inference', dest='enableinference', type=bool_t, default='True', help='Enable Inference during training (Consumes 2GB of VRAM)')
parser.add_argument('--extended_validation', type=bool_t, default='False', help='Perform extended validation of images to catch truncated or corrupt images.')
parser.add_argument('--no_migration', type=bool_t, default='False', help='Do not perform migration of dataset while the `--resize` flag is active. Migration creates an adjacent folder to the dataset with <dataset_dirname>_cropped.')
parser.add_argument('--skip_validation', type=bool_t, default='False', help='Skip validation of images, useful for speeding up loading of very large datasets that have already been validated.')
parser.add_argument('--extended_mode_chunks', type=int, default=3, help='Enables extended mode for tokenization with given amount of maximum chunks. Values < 2 disable.')
parser.add_argument("--train_text_encoder", action="store_true", help="Whether to train the text encoder")
parser.add_argument(
        "--snr_gamma",
        type=float,
        default=5.0,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )

args = parser.parse_args()

def setup():
    try :
        torch.distributed.init_process_group("nccl", init_method="env://")
        print('distributed training is ENABLED')
        return True
    except Exception :
        print('distributed training is DISABLED')
        return False

def cleanup():
    torch.distributed.destroy_process_group()

def get_rank() -> int:
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()

def get_world_size() -> int:
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()

def get_gpu_ram() -> str:
    """
    Returns memory usage statistics for the CPU, GPU, and Torch.

    :return:
    """
    gpu_str = ""
    torch_str = ""
    try:
        cudadev = torch.cuda.current_device()
        nvml_device = pynvml.nvmlDeviceGetHandleByIndex(cudadev)
        gpu_info = pynvml.nvmlDeviceGetMemoryInfo(nvml_device)
        gpu_total = int(gpu_info.total / 1E6)
        gpu_free = int(gpu_info.free / 1E6)
        gpu_used = int(gpu_info.used / 1E6)
        gpu_str = f"GPU: (U: {gpu_used:,}mb F: {gpu_free:,}mb " \
                  f"T: {gpu_total:,}mb) "
        torch_reserved_gpu = int(torch.cuda.memory.memory_reserved() / 1E6)
        torch_reserved_max = int(torch.cuda.memory.max_memory_reserved() / 1E6)
        torch_used_gpu = int(torch.cuda.memory_allocated() / 1E6)
        torch_max_used_gpu = int(torch.cuda.max_memory_allocated() / 1E6)
        torch_str = f"TORCH: (R: {torch_reserved_gpu:,}mb/"  \
                    f"{torch_reserved_max:,}mb, " \
                    f"A: {torch_used_gpu:,}mb/{torch_max_used_gpu:,}mb)"
    except AssertionError:
        pass
    cpu_maxrss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1E3 +
                     resource.getrusage(
                         resource.RUSAGE_CHILDREN).ru_maxrss / 1E3)
    cpu_vmem = psutil.virtual_memory()
    cpu_free = int(cpu_vmem.free / 1E6)
    return f"CPU: (maxrss: {cpu_maxrss:,}mb F: {cpu_free:,}mb) " \
           f"{gpu_str}" \
           f"{torch_str}"

def _sort_by_ratio(bucket: tuple) -> float:
    return bucket[0] / bucket[1]

def _sort_by_area(bucket: tuple) -> float:
    return bucket[0] * bucket[1]

class Validation():
    def __init__(self, is_skipped: bool, is_extended: bool) -> None:
        if is_skipped:
            self.validate = self.__no_op
            return print("Validation: Skipped")

        if is_extended:
            self.validate = self.__extended_validate
            return print("Validation: Extended")

        self.validate = self.__validate
        print("Validation: Standard")

    def __validate(self, fp: str) -> bool:
        try:
            img = Image.open(fp)
            [s, _] = os.path.splitext(fp)
            return img is not None and os.path.exists(s + '.txt')
        except:
            print(f'WARNING: Image cannot be opened: {fp}')
            return False

    def __extended_validate(self, fp: str) -> bool:
        try:
            Image.open(fp).load()
            return True
        except (OSError) as error:
            if 'truncated' in str(error):
                print(f'WARNING: Image truncated: {error}')
                return False
            print(f'WARNING: Image cannot be opened: {error}')
            return False
        except:
            print(f'WARNING: Image cannot be opened: {error}')
            return False

    def __no_op(self, fp: str) -> bool:
        return True

class Resize():
    def __init__(self, is_resizing: bool, is_not_migrating: bool) -> None:
        if not is_resizing:
            self.resize = self.__no_op
            return

        if not is_not_migrating:
            self.resize = self.__migration
            dataset_path = os.path.split(args.dataset)
            self.__directory = os.path.join(
                dataset_path[0],
                f'{dataset_path[1]}_cropped'
            )
            os.makedirs(self.__directory, exist_ok=True)
            return print(f"Resizing: Performing migration to '{self.__directory}'.")

        self.resize = self.__no_migration

    def __no_migration(self, image_path: str, w: int, h: int) -> Img:
        return ImageOps.fit(
                Image.open(image_path),
                (w, h),
                bleed=0.0,
                centering=(0.5, 0.5),
                method=Image.Resampling.LANCZOS
            )

    def __migration(self, image_path: str, w: int, h: int) -> Img:
        filename = re.sub('\.[^/.]+$', '', os.path.split(image_path)[1])

        image = ImageOps.fit(
                Image.open(image_path),
                (w, h),
                bleed=0.0,
                centering=(0.5, 0.5),
                method=Image.Resampling.LANCZOS
            )

        image.save(
            os.path.join(f'{self.__directory}', f'{filename}.jpg'),
            optimize=True
        )

        try:
            shutil.copy(
                os.path.join(args.dataset, f'{filename}.txt'),
                os.path.join(self.__directory, f'{filename}.txt'),
                follow_symlinks=False
            )
        except (FileNotFoundError):
            f = open(
                os.path.join(self.__directory, f'{filename}.txt'),
                'w',
                encoding='UTF-8'
            )
            f.close()

        return image

    def __no_op(self, image_path: str, w: int, h: int) -> Img:
        return Image.open(image_path)

class ImageStore:
    def __init__(self, data_dirs: str) -> None:
        self.data_dirs = data_dirs.split(',')
        data_dir = data_dirs
        for data_dir in self.data_dirs :
            print('include', data_dir)

        exts = ['jpg', 'jpeg', 'png', 'bmp', 'webp']
        exts = ['jpg']

        self.image_files = []
        for data_dir in self.data_dirs :
            print('listing files in', data_dir)
            [self.image_files.extend(glob.glob(f'{data_dir}' + '/*.' + e)[:100000000]) for e in exts]
            [self.image_files.extend(glob.glob(f'{data_dir}' + '/**/*.' + e)[:100000000]) for e in exts]
        self.image_files = list(set(self.image_files))#[:1000]

        # self.image_files = []
        # #[self.image_files.extend(glob.glob(f'{data_dir}' + '/**/*.' + e)) for e in ['jpg', 'jpeg', 'png', 'bmp', 'webp']]
        # for i in range(20) :
        #     folder = '%05d' % i
        #     [self.image_files.extend(glob.glob(f'{data_dir}' + f'/{folder}/*.' + e)) for e in ['jpg']]
        # [self.image_files.extend(glob.glob(f'{data_dir}' + '/**/*.' + e)) for e in ['jpg']]

        self.validator = Validation(
            args.skip_validation,
            args.extended_validation
        ).validate

        self.resizer = Resize(args.resize, args.no_migration).resize

        print(' -- before validation we have', len(self.image_files), 'images')
        self.image_files = [x for x in self.image_files if self.validator(x)]
        print(' -- after validation we have', len(self.image_files), 'images')

        self.caption_cache = {}

    def __len__(self) -> int:
        return len(self.image_files)

    # iterator returns images as PIL images and their index in the store
    def entries_iterator(self) -> Generator[Tuple[Img, int], None, None]:
        for f in range(len(self)):
            yield Image.open(self.image_files[f]), f

    def caption_iterator(self) -> Generator[Tuple[str, int, str], None, None]:
        for f in range(len(self)):
            filename, file_extension = os.path.splitext(self.image_files[f])
            if os.path.exists(filename + '.txt') :
                with open(filename + '.txt', 'r', encoding='UTF-8') as fp:
                    txt = fp.read()
            else :
                txt = ''
            yield txt, f, self.image_files[f]

    # get image by index
    def get_image(self, ref: Tuple[int, int, int]) -> Img:
        img = Image.open(self.image_files[ref[0]]).convert('RGB')
        w, h = img.size
        bucket_w = ref[1]
        bucket_h = ref[2]
        if w / h >= bucket_w / bucket_h :
            # cut width
            sheight = bucket_h
            swidth = int(round(w / (h / bucket_h)))
        else :
            # cut height
            swidth = bucket_w
            sheight = int(round(h / (w / bucket_w)))
        img2 = img.resize((swidth, sheight), resample = Image.Resampling.BICUBIC)
        crop_params = transforms.RandomCrop.get_params(img2, (bucket_h, bucket_w))
        img3 = visionF.crop(img2, *crop_params)
        del img
        del img2
        return img3

    # gets caption by removing the extension from the filename and replacing it with .txt
    def get_caption(self, ref: Tuple[int, int, int]) -> str:
        if ref[0] in self.caption_cache :
            return self.caption_cache[ref[0]]
        else :
            filename, file_extension = os.path.splitext(self.image_files[ref[0]])
            try :
                if os.path.exists(filename + '.txt') :
                    with open(filename + '.txt', 'r', encoding='UTF-8') as fp:
                        self.caption_cache[ref[0]] = fp.read()
                else :
                    txt = ''
            except Exception :
                return ''
            return self.caption_cache[ref[0]]


# ====================================== #
# Bucketing code stolen from hasuwoof:   #
# https://github.com/hasuwoof/huskystack #
# ====================================== #

from dataclasses import dataclass

@dataclass
class TagFreqAdjust :
    tag: str
    adjustment: Optional[float]
    forced_prob: Optional[float]
    allow_decrement: Optional[bool] = False

class AspectBucket:
    def __init__(self, store: ImageStore,
                 num_buckets: int,
                 batch_size: int,
                 bucket_side_min: int = 256,
                 bucket_side_max: int = 1536,
                 bucket_side_increment: int = 64,
                 max_image_area: int = 512 * 768,
                 max_ratio: float = 3,
                 freq_adjust: List[TagFreqAdjust] = []):

        self.requested_bucket_count = num_buckets
        self.bucket_length_min = bucket_side_min
        self.bucket_length_max = bucket_side_max
        self.bucket_increment = bucket_side_increment
        self.max_image_area = max_image_area
        self.batch_size = batch_size
        self.total_dropped = 0
        self.freq_adjust = freq_adjust
        print(freq_adjust)

        if max_ratio <= 0:
            self.max_ratio = float('inf')
        else:
            self.max_ratio = max_ratio

        self.store = store
        self.buckets = []
        self._bucket_ratios = []
        self._bucket_interp = None
        self.bucket_data: Dict[tuple, List[int]] = dict()
        self.init_buckets()
        self.total_images = 0
        self.tag_freq_map = defaultdict(int)
        self.tag_image_index_map = defaultdict(list)
        self.freq_adjusted_image_store_indices = []
        self.perform_freq_asjustment()
        self.fill_buckets()

    def perform_freq_asjustment(self) :
        entries = self.store.caption_iterator()
        print('performing tag frequency adjustment')
        for caption, idx, image_filename in tqdm.tqdm(entries) :
            self.total_images += 1
            tags = caption.split(',')
            for tag in tags :
                self.tag_freq_map[tag] += 1
                self.tag_image_index_map[tag].append(idx)
        adjusted_image_indices = set()
        all_image_indices = set(range(self.total_images))
        for adjustment in self.freq_adjust :
            tag = adjustment.tag
            old_freq = self.tag_freq_map[adjustment.tag] / self.total_images
            if adjustment.forced_prob is not None :
                new_freq = adjustment.forced_prob
            elif adjustment.adjustment is not None :
                new_freq = adjustment * old_freq
            image_diff = int((new_freq * self.total_images - self.tag_freq_map[adjustment.tag]) / (1 - new_freq))
            image_indices = copy.deepcopy(self.tag_image_index_map[adjustment.tag])
            if not image_indices :
                print(' -- warn, empty', adjustment.tag)
                continue
            adjusted_image_indices.update(image_indices)
            if image_diff > 0 :
                extra_image_indices = np.random.choice(image_indices, image_diff, replace = True)
                image_indices.extend(extra_image_indices)
            elif image_diff < 0 and adjustment.allow_decrement :
                np.random.shuffle(image_indices)
                image_indices = image_indices[: image_diff]
            print('Tag adjustment for', adjustment.tag, 'is from', self.tag_freq_map[adjustment.tag], 'with', image_diff, 'to', len(image_indices))
            self.freq_adjusted_image_store_indices.extend(image_indices)
        self.freq_adjusted_image_store_indices.extend(list(all_image_indices - adjusted_image_indices))
        np.random.shuffle(self.freq_adjusted_image_store_indices)

    def iterate_frequency_adjusted_images(self) -> Generator[Tuple[Img, int], None, None]:
        for idx in range(len(self.freq_adjusted_image_store_indices)):
            image_store_index = self.freq_adjusted_image_store_indices[idx]
            yield Image.open(self.store.image_files[image_store_index]), image_store_index

    def init_buckets(self):
        possible_lengths = list(range(self.bucket_length_min, self.bucket_length_max + 1, self.bucket_increment))
        possible_buckets = list((w, h) for w, h in itertools.product(possible_lengths, possible_lengths)
                        if w >= h and w * h <= self.max_image_area and w / h <= self.max_ratio)

        buckets_by_ratio = {}

        # group the buckets by their aspect ratios
        for bucket in possible_buckets:
            w, h = bucket
            # use precision to avoid spooky floats messing up your day
            ratio = '{:.4e}'.format(w / h)

            if ratio not in buckets_by_ratio:
                group = set()
                buckets_by_ratio[ratio] = group
            else:
                group = buckets_by_ratio[ratio]

            group.add(bucket)

        # now we take the list of buckets we generated and pick the largest by area for each (the first sorted)
        # then we put all of those in a list, sorted by the aspect ratio
        # the square bucket (LxL) will be the first
        unique_ratio_buckets = sorted([sorted(buckets, key=_sort_by_area)[-1]
                                       for buckets in buckets_by_ratio.values()], key=_sort_by_ratio)

        # how many buckets to create for each side of the distribution
        bucket_count_each = int(np.clip((self.requested_bucket_count + 1) / 2, 1, len(unique_ratio_buckets)))

        # we know that the requested_bucket_count must be an odd number, so the indices we calculate
        # will include the square bucket and some linearly spaced buckets along the distribution
        indices = {*np.linspace(0, len(unique_ratio_buckets) - 1, bucket_count_each, dtype=int)}

        # make the buckets, make sure they are unique (to remove the duplicated square bucket), and sort them by ratio
        # here we add the portrait buckets by reversing the dimensions of the landscape buckets we generated above
        buckets = sorted({*(unique_ratio_buckets[i] for i in indices),
                          *(tuple(reversed(unique_ratio_buckets[i])) for i in indices)}, key=_sort_by_ratio)

        self.buckets = buckets

        # cache the bucket ratios and the interpolator that will be used for calculating the best bucket later
        # the interpolator makes a 1d piecewise interpolation where the input (x-axis) is the bucket ratio,
        # and the output is the bucket index in the self.buckets array
        # to find the best fit we can just round that number to get the index
        self._bucket_ratios = [w / h for w, h in buckets]
        self._bucket_interp = interp1d(self._bucket_ratios, list(range(len(buckets))), assume_sorted=True,
                                       fill_value=None)

        for b in buckets:
            self.bucket_data[b] = []

    def get_batch_count(self):
        return sum(len(b) // self.batch_size for b in self.bucket_data.values())

    def get_bucket_info(self):
        return json.dumps({ "buckets": self.buckets, "bucket_ratios": self._bucket_ratios })

    def get_batch_iterator(self) -> Generator[Tuple[Tuple[int, int, int]], None, None]:
        """
        Generator that provides batches where the images in a batch fall on the same bucket

        Each element generated will be:
            (index, w, h)

        where each image is an index into the dataset
        :return:
        """
        max_bucket_len = max(len(b) for b in self.bucket_data.values())
        index_schedule = list(range(max_bucket_len))
        random.shuffle(index_schedule)

        bucket_len_table = {
            b: len(self.bucket_data[b]) for b in self.buckets
        }

        bucket_schedule = []
        for i, b in enumerate(self.buckets):
            bucket_schedule.extend([i] * (bucket_len_table[b] // self.batch_size))

        random.shuffle(bucket_schedule)

        bucket_pos = {
            b: 0 for b in self.buckets
        }

        total_generated_by_bucket = {
            b: 0 for b in self.buckets
        }

        for bucket_index in bucket_schedule:
            b = self.buckets[bucket_index]
            i = bucket_pos[b]
            bucket_len = bucket_len_table[b]

            batch = []
            while len(batch) != self.batch_size:
                # advance in the schedule until we find an index that is contained in the bucket
                k = index_schedule[i]
                if k < bucket_len:
                    entry = self.bucket_data[b][k]
                    batch.append(entry)

                i += 1

            total_generated_by_bucket[b] += self.batch_size
            bucket_pos[b] = i
            yield [(idx, *b) for idx in batch]

    def fill_buckets(self):
        entries = self.iterate_frequency_adjusted_images()
        total_dropped = 0

        print('performing bucket construction')
        for entry, index in tqdm.tqdm(entries, total=len(self.freq_adjusted_image_store_indices)):
            if not self._process_entry(entry, index):
                total_dropped += 1

        for b, values in self.bucket_data.items():
            # shuffle the entries for extra randomness and to make sure dropped elements are also random
            random.shuffle(values)

            # make sure the buckets have an exact number of elements for the batch
            to_drop = len(values) % self.batch_size
            self.bucket_data[b] = list(values[:len(values) - to_drop])
            total_dropped += to_drop

        self.total_dropped = total_dropped

    def _process_entry(self, entry: Image.Image, index: int) -> bool:
        aspect = entry.width / entry.height

        if aspect > self.max_ratio or (1 / aspect) > self.max_ratio:
            return False

        best_bucket = self._bucket_interp(aspect)

        if best_bucket is None:
            return False


        bucket = self.buckets[round(float(best_bucket))]
        (bucket_w, bucket_h) = bucket

        w, h = entry.size
        if w / h >= bucket_w / bucket_h :
            # cut width
            sheight = bucket_h
            swidth = int(round(w / (h / bucket_h)))
            if (swidth - bucket_w) / bucket_w > 0.1 :
                return False
        else :
            # cut height
            swidth = bucket_w
            sheight = int(round(h / (w / bucket_w)))
            if (sheight - bucket_h) / bucket_h > 0.1 :
                return False


        self.bucket_data[bucket].append(index)

        del entry

        return True

class AspectBucketSampler(torch.utils.data.Sampler):
    def __init__(self, bucket: AspectBucket, num_replicas: int = 1, rank: int = 0):
        super().__init__(None)
        self.bucket = bucket
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        # subsample the bucket to only include the elements that are assigned to this rank
        indices = self.bucket.get_batch_iterator()
        indices = list(indices)[self.rank::self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.bucket.get_batch_count() // self.num_replicas


def process_tags(tags: List[str], min_tags=24, max_tags=70):
    tags = [tag.strip() for tag in tags]
    if np.random.randint(0, 100) < 50 :
        tags = [tag.replace('_', ' ') for tag in tags]
    final_tags = []
    for tag in tags :
        if ':' in tag :
            tag = tag.split(':')[-1]
        final_tags.append(tag)
    kept_tags = np.random.randint(min_tags, max_tags + 1)
    np.random.shuffle(final_tags)
    final_tags = final_tags[:kept_tags]
    return final_tags, False

def expand_prefix(tag: str) -> str :
    if ':' in tag :
        if tag.startswith("art:"):
            tag = 'artist' + tag[3:]
        elif tag.startswith("copy:"):
            tag = 'copyright' + tag[4:]
        elif tag.startswith("char:"):
            tag = 'character' + tag[4:]
        elif tag.startswith("gen:"):
            tag = 'general' + tag[3:]
        return tag
    else :
        return tag

def strip_prefix(tag: str) -> str :
    if ':' in tag :
        return tag.split(':')[-1]
    else :
        return tag

def is_artist_or_character(tag):
    return tag.startswith("character:") or tag.startswith("artist:")

class AspectDataset(torch.utils.data.Dataset):
    def __init__(self, store: ImageStore, device: torch.device, tokenizer1, tokenizer2, ucg: float = 0.1):
        self.store = store
        self.device = device
        self.ucg = ucg
        self.tokenizer1 = tokenizer1
        self.tokenizer2 = tokenizer2

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5])
        ])
        

    def process_prompt(self, tags: str) -> Tuple[str, float] :
        tags = tags.split(',')
        tags = [expand_prefix(t) for t in tags]
        tags, _ = process_tags(tags)
        np.random.shuffle(tags)
        return ','.join(tags), 1

    def __len__(self):
        return len(self.store)

    def __getitem__(self, item: Tuple[int, int, int]):
        return_dict = {'pixel_values': None, 'prompt': None, 'weight': None}

        image_file = self.store.get_image(item)

        return_dict['pixel_values'] = self.transforms(image_file)
        prompt, weight = self.process_prompt(self.store.get_caption(item))
        return_dict['weight'] = weight
        if random.random() > self.ucg:
            pass
        else:
            prompt = ''

        return_dict['prompt'] = prompt
        return return_dict

    def collate_fn(self, examples):
        pixel_values = torch.stack([example['pixel_values'] for example in examples if example is not None])
        pixel_values.to(memory_format=torch.contiguous_format).float()

        max_length = self.tokenizer1.model_max_length
        max_chunks = args.extended_mode_chunks

        prompts = [example['prompt'] for example in examples]
        input_ids1 = [self.tokenizer1([prompt], truncation=True, return_length=True, return_overflowing_tokens=False, padding=False, add_special_tokens=False, max_length=(max_length * max_chunks) - (max_chunks * 2)).input_ids[0] for prompt in prompts if prompt is not None]
        input_ids2 = [self.tokenizer2([prompt], truncation=True, return_length=True, return_overflowing_tokens=False, padding=False, add_special_tokens=False, max_length=(max_length * max_chunks) - (max_chunks * 2)).input_ids[0] for prompt in prompts if prompt is not None]

        weights = torch.tensor([example['weight'] for example in examples], dtype = torch.float32)
        return {
            'pixel_values': pixel_values,
            'prompts': prompts,
            'input_ids1': input_ids1,
            'input_ids2': input_ids2,
            'weights': weights
        }

def encode_prompts_small_clip(device, tokenizer, text_encoder, input_ids) :
    if type(text_encoder) is torch.nn.parallel.DistributedDataParallel:
        text_encoder = text_encoder.module
    max_length = tokenizer.model_max_length
    #max_chunks = args.extended_mode_chunks
    #input_ids = [tokenizer([prompt], truncation=True, return_length=True, return_overflowing_tokens=False, padding=False, add_special_tokens=False, max_length=(max_length * max_chunks) - (max_chunks * 2)).input_ids[0] for prompt in prompts if prompt is not None]

    args.clip_penultimate = True
    layer_idx = -2 if args.clip_penultimate else -1

    with torch.autocast('cuda', enabled=args.fp16):
        max_standard_tokens = max_length - 2
        max_chunks = args.extended_mode_chunks
        max_len = np.ceil(max(len(x) for x in input_ids) / max_standard_tokens).astype(int).item() * max_standard_tokens
        if max_len > max_standard_tokens:
            for i, x in enumerate(input_ids):
                if len(x) < max_len:
                    input_ids[i] = [*x, *np.full((max_len - len(x)), tokenizer.eos_token_id)]
            batch_t = torch.tensor(input_ids)
            chunks = [batch_t[:, i:i + max_standard_tokens] for i in range(0, max_len, max_standard_tokens)]
            chunk_result = list(range(len(chunks)))
            for i, chunk in enumerate(chunks):
                chunk = torch.cat((torch.full((chunk.shape[0], 1), tokenizer.bos_token_id), chunk, torch.full((chunk.shape[0], 1), tokenizer.eos_token_id)), 1)
                chunk_result[i] = text_encoder(chunk.to(device), output_hidden_states=True)['hidden_states'][layer_idx]
            outs = torch.cat(chunk_result, dim=-2)
        else:
            for i, x in enumerate(input_ids):
                input_ids[i] = [tokenizer.bos_token_id, *x, *np.full((tokenizer.model_max_length - len(x) - 1), tokenizer.eos_token_id)]
            outs = text_encoder(torch.asarray(input_ids).to(device), output_hidden_states=True)['hidden_states'][layer_idx]
    outs = torch.stack(tuple(outs))
    return outs

def encode_prompts_big_clip(device, tokenizer, text_encoder, input_ids) :
    if type(text_encoder) is torch.nn.parallel.DistributedDataParallel:
        text_encoder = text_encoder.module
    max_length = tokenizer.model_max_length
    #max_chunks = args.extended_mode_chunks
    #input_ids = [tokenizer([prompt], truncation=True, return_length=True, return_overflowing_tokens=False, padding=False, add_special_tokens=False, max_length=(max_length * max_chunks) - (max_chunks * 2)).input_ids[0] for prompt in prompts if prompt is not None]

    args.clip_penultimate = True
    layer_idx = -2 if args.clip_penultimate else -1

    all_pool_outputs = []
    with torch.autocast('cuda', enabled=args.fp16):
        max_standard_tokens = max_length - 2
        max_chunks = args.extended_mode_chunks
        max_len = np.ceil(max(len(x) for x in input_ids) / max_standard_tokens).astype(int).item() * max_standard_tokens
        if max_len > max_standard_tokens:
            for i, x in enumerate(input_ids):
                if len(x) < max_len:
                    input_ids[i] = [*x, *np.full((max_len - len(x)), tokenizer.eos_token_id)]
            batch_t = torch.tensor(input_ids)
            chunks = [batch_t[:, i:i + max_standard_tokens] for i in range(0, max_len, max_standard_tokens)]
            chunk_result = list(range(len(chunks)))
            for i, chunk in enumerate(chunks):
                chunk = torch.cat((torch.full((chunk.shape[0], 1), tokenizer.bos_token_id), chunk, torch.full((chunk.shape[0], 1), tokenizer.eos_token_id)), 1)
                out_states = text_encoder(chunk.to(device), output_hidden_states=True)
                text_embeds = out_states.text_embeds # pooled
                all_pool_outputs.append(text_embeds)
                chunk_result[i] = out_states.hidden_states[layer_idx]
            outs = torch.cat(chunk_result, dim=-2)
        else:
            for i, x in enumerate(input_ids):
                input_ids[i] = [tokenizer.bos_token_id, *x, *np.full((tokenizer.model_max_length - len(x) - 1), tokenizer.eos_token_id)]
            out_states = text_encoder(torch.asarray(input_ids).to(device), output_hidden_states=True)
            text_embeds = out_states.text_embeds # pooled
            all_pool_outputs.append(text_embeds)
            outs = out_states.hidden_states[layer_idx]
    outs = torch.stack(tuple(outs))
    pooled_output = torch.stack(all_pool_outputs, dim = -1).mean(-1) # average of pooled text embd
    return outs, pooled_output

# Adapted from torch-ema https://github.com/fadel/pytorch_ema/blob/master/torch_ema/ema.py#L14
class EMAModel:
    """
    Exponential Moving Average of models weights
    """

    def __init__(self, parameters: Iterable[torch.nn.Parameter], decay=0.9999):
        parameters = list(parameters)
        self.shadow_params = [p.clone().detach() for p in parameters]

        self.decay = decay
        self.optimization_step = 0

    def get_decay(self, optimization_step):
        """
        Compute the decay factor for the exponential moving average.
        """
        value = (1 + optimization_step) / (10 + optimization_step)
        return 1 - min(self.decay, value)

    @torch.no_grad()
    def step(self, parameters):
        parameters = list(parameters)

        self.optimization_step += 1
        self.decay = self.get_decay(self.optimization_step)

        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                tmp = self.decay * (s_param - param)
                s_param.sub_(tmp)
            else:
                s_param.copy_(param)

        torch.cuda.empty_cache()

    def copy_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """
        Copy current averaged parameters into given collection of parameters.
        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = list(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.data)

    # From CompVis LitEMA implementation
    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

        del self.collected_params
        gc.collect()

    def to(self, device=None, dtype=None) -> None:
        r"""Move internal buffers of the ExponentialMovingAverage to `device`.
        Args:
            device: like `device` argument to `torch.Tensor.to`
        """
        # .to() on the tensors handles None correctly
        self.shadow_params = [
            p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device)
            for p in self.shadow_params
        ]

def _get_add_time_ids(unet, text_encoder_2, original_size, crops_coords_top_left, target_size, dtype):
    add_time_ids = list(original_size + crops_coords_top_left + target_size)

    try :
        passed_add_embed_dim = (
            unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_2.config.projection_dim
        )
        expected_add_embed_dim = unet.add_embedding.linear_1.in_features
    except AttributeError :
        passed_add_embed_dim = (
            unet.module.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_2.module.config.projection_dim
        )
        expected_add_embed_dim = unet.module.add_embedding.linear_1.in_features

    if expected_add_embed_dim != passed_add_embed_dim:
        raise ValueError(
            f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
        )

    add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
    return add_time_ids

def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr

def main(enabled_dis = True):
    """
    TODO:
        better image loader
        gradient accumulation
        //tag loss manager
        //randomize tags
        //sd-webui text encoding
        prior perserving loss
        inpainting objective
    """
    rank = get_rank()
    world_size = get_world_size()
    torch.cuda.set_device(rank)

    if rank == 0:
        os.makedirs(args.output_path, exist_ok=True)
        
        mode = 'disabled'
        if args.enablewandb:
            mode = 'online'
        if args.hf_token is not None:
            os.environ['HF_API_TOKEN'] = args.hf_token
            args.hf_token = None
        run = wandb.init(project=args.project_id, name=args.run_name, config=vars(args), dir=args.output_path+'/wandb', mode=mode)

        # Inform the user of host, and various versions -- useful for debugging issues.
        print("RUN_NAME:", args.run_name)
        print("HOST:", socket.gethostname())
        print("CUDA:", torch.version.cuda)
        print("TORCH:", torch.__version__)
        print("TRANSFORMERS:", transformers.__version__)
        print("DIFFUSERS:", diffusers.__version__)
        print("MODEL:", args.model)
        print("FP16:", args.fp16)
        print("RESOLUTION:", args.resolution)
        print("BATCH_SIZE:", args.batch_size)
        print('Using fp32 for VAE')


    if args.hf_token is not None:
        print('It is recommended to set the HF_API_TOKEN environment variable instead of passing it as a command line argument since WandB will automatically log it.')
    else:
        try:
            args.hf_token = os.environ['HF_API_TOKEN']
            print("HF Token set via enviroment variable")
        except Exception:
            print("No HF Token detected in arguments or enviroment variable, setting it to none (as in string)")
            args.hf_token = "none"

    device = torch.device('cuda')

    print("DEVICE:", device)

    # setup fp16 stuff
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    # Set seed
    # torch.manual_seed(args.seed + rank)
    # random.seed(args.seed + rank)
    # np.random.seed(args.seed + rank)
    # print('RANDOM SEED:', args.seed + rank)
    # torch.manual_seed(args.seed)
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # print('RANDOM SEED:', args.seed)

    if args.resume:
        args.model = args.resume
    
    tokenizer1 = CLIPTokenizer.from_pretrained(os.path.join(args.model, 'tokenizer'), use_auth_token=args.hf_token)
    text_encoder1 = CLIPTextModel.from_pretrained(os.path.join(args.model, 'text_encoder'), use_auth_token=args.hf_token)
    tokenizer2 = CLIPTokenizer.from_pretrained(os.path.join(args.model, 'tokenizer_2'), use_auth_token=args.hf_token)
    text_encoder2 = CLIPTextModelWithProjection.from_pretrained(os.path.join(args.model, 'text_encoder_2'), use_auth_token=args.hf_token)
    vae = AutoencoderKL.from_pretrained(os.path.join(args.model, 'vae'), use_auth_token=args.hf_token)
    unet = UNet2DConditionModel.from_pretrained(os.path.join(args.model, 'unet'), use_auth_token=args.hf_token)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    if not args.train_text_encoder:
        text_encoder1.requires_grad_(False)
        text_encoder2.requires_grad_(False)
    else :
        text_encoder1.requires_grad_(True)
        text_encoder2.requires_grad_(True)

    unet.enable_gradient_checkpointing()
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder1.gradient_checkpointing_enable()
            text_encoder2.gradient_checkpointing_enable()

    args.use_xformers = True
    if args.use_xformers:
        unet.set_use_memory_efficient_attention_xformers(True)

    # "The “safer” approach would be to move the model to the device first and create the optimizer afterwards."
    weight_dtype = torch.float16 if args.fp16 else torch.float32

    # move models to device
    vae = vae.to(device, dtype=torch.float32)
    vae.eval()
    unet = unet.to(device, dtype=torch.float32)
    text_encoder1 = text_encoder1.to(device, dtype=weight_dtype if not args.train_text_encoder else torch.float32)
    text_encoder2 = text_encoder2.to(device, dtype=weight_dtype if not args.train_text_encoder else torch.float32)

    if enabled_dis :
        unet = torch.nn.parallel.DistributedDataParallel(
            unet,
            device_ids=[rank],
            output_device=rank,
            gradient_as_bucket_view=True
        )

    if enabled_dis :
        if args.train_text_encoder:
            text_encoder1 = torch.nn.parallel.DistributedDataParallel(
                text_encoder1,
                device_ids=[rank],
                output_device=rank,
                gradient_as_bucket_view=True
            )
            text_encoder2 = torch.nn.parallel.DistributedDataParallel(
                text_encoder2,
                device_ids=[rank],
                output_device=rank,
                gradient_as_bucket_view=True
            )

    if args.use_8bit_adam: # Bits and bytes is only supported on certain CUDA setups, so default to regular adam if it fails.
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        except:
            print('bitsandbytes not supported, using regular Adam optimizer')
            optimizer_cls = torch.optim.AdamW
    else:
        optimizer_cls = torch.optim.AdamW

    """
    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
        weight_decay=args.adam_weight_decay,
    )
    """

    optimizer_parameters = unet.parameters() if not args.train_text_encoder else itertools.chain(unet.parameters(), text_encoder1.parameters(), text_encoder2.parameters())

    # Create distributed optimizer
    if enabled_dis :
        from torch.distributed.optim import ZeroRedundancyOptimizer
        optimizer = ZeroRedundancyOptimizer(
            optimizer_parameters,
            optimizer_class=optimizer_cls,
            parameters_as_bucket_view=True,
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )
    else :
        optimizer = optimizer_cls(
            unet.parameters(),
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )


    noise_scheduler = EulerDiscreteScheduler.from_pretrained(
        args.model,
        subfolder='scheduler',
        use_auth_token=args.hf_token,
    )

    freq_adjs = []

    # load dataset
    store = ImageStore(args.dataset)
    dataset = AspectDataset(store, device, ucg=args.ucg, tokenizer1 = tokenizer1, tokenizer2 = tokenizer2)
    bucket = AspectBucket(store, args.num_buckets, args.batch_size, args.bucket_side_min, args.bucket_side_max, 64, args.resolution * args.resolution, 3.0, freq_adjust=freq_adjs)
    sampler = AspectBucketSampler(bucket=bucket, num_replicas=world_size, rank=rank)

    print(f'STORE_LEN: {len(store)}')
    print(bucket.get_bucket_info())

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,
        collate_fn=dataset.collate_fn
    )
    
    # Migrate dataset
    if args.resize and not args.no_migration:
        for _, batch in enumerate(train_dataloader):
            continue
        print(f"Completed resize and migration to '{args.dataset}_cropped' please relaunch the trainer without the --resize argument and train on the migrated dataset.")
        exit(0)

    # create ema
    if args.use_ema:
        ema_unet = EMAModel(unet.parameters())

    print(get_gpu_ram())

    num_steps_per_epoch = len(train_dataloader)
    progress_bar = tqdm.tqdm(range(args.epochs * num_steps_per_epoch), desc="Total Steps", leave=False)
    global_step = 0

    if args.resume:
        target_global_step = int(args.resume.split('_')[-1])
        print(f'resuming from {args.resume}...')

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=int(args.lr_scheduler_warmup * num_steps_per_epoch * args.epochs),
        num_training_steps=args.epochs * num_steps_per_epoch,
        #last_epoch=(global_step // num_steps_per_epoch) - 1,
    )

    def save_checkpoint(global_step):
        if rank == 0:
            if args.use_ema:
                ema_unet.store(unet.parameters())
                ema_unet.copy_to(unet.parameters())
            pipeline = StableDiffusionXLPipeline(
                text_encoder=text_encoder1 if type(text_encoder1) is not torch.nn.parallel.DistributedDataParallel else text_encoder1.module,
                text_encoder_2=text_encoder2 if type(text_encoder2) is not torch.nn.parallel.DistributedDataParallel else text_encoder2.module,
                vae=vae,
                unet=unet if type(unet) is not torch.nn.parallel.DistributedDataParallel else unet.module,
                tokenizer=tokenizer1,
                tokenizer_2=tokenizer2,
                scheduler=EulerDiscreteScheduler.from_pretrained(args.model, subfolder="scheduler", use_auth_token=args.hf_token),
            )
            print(f'saving checkpoint to: {args.output_path}/{args.run_name}_{global_step}')
            pipeline.save_pretrained(f'{args.output_path}/{args.run_name}_{global_step}')

            if args.use_ema:
                ema_unet.restore(unet.parameters())

    # train!
    gas = args.gradient_accumulation
    snr_gamma = args.snr_gamma
    load_time_ema = 0
    seconds_per_step_ema = 0
    start_time = time.time()
    vae_b_ema = 0
    text_encode_b_ema = 0
    noise_sch_b_ema = 0
    noise_pred_b_ema = 0
    opt_b_ema = 0
    ema_b_ema = 0
    reduce_b_ema = 0
    other_b_ema = 0
    try:
        loss = torch.tensor(0.0, device=device, dtype=weight_dtype)
        local_counter = 0
        for epoch in range(args.epochs):
            unet.train()
            if args.train_text_encoder:
                text_encoder1.train()
                text_encoder2.train()
            optimizer.zero_grad()
            for _, batch in enumerate(train_dataloader):
                end_time = time.time()
                load_time_ema = 0.9 * load_time_ema + 0.1 * (end_time - start_time)

                batch_weights = batch['weights'].to(device, dtype=weight_dtype)
                
                b_start = time.perf_counter()
                vae_b_start = time.perf_counter()
                with torch.no_grad(), torch.autocast('cuda', enabled=False) :
                    latents = vae.encode(batch['pixel_values'].to(device, dtype=torch.float32)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                    # Sample noise
                    noise = torch.randn_like(latents) + 0.1 * torch.randn(latents.shape[0], latents.shape[1], 1, 1, device = latents.device)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()

                    # betas = np.random.beta(0.76, 1.16, size = (bsz)).astype(np.float32)
                    # timesteps = np.round(betas * (noise_scheduler.num_train_timesteps - 1))
                    # timesteps = torch.from_numpy(timesteps).long().to(latents.device)

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                vae_b_end = time.perf_counter()

                text_encode_b_start = time.perf_counter()
                # Get the embedding for conditioning
                encoder_hidden_states1 = encode_prompts_small_clip(device, tokenizer1, text_encoder1, batch['input_ids1'])
                encoder_hidden_states2, add_text_embeds = encode_prompts_big_clip(device, tokenizer2, text_encoder2, batch['input_ids2'])
                encoder_hidden_states = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim = 2) # 1x77x(768+1280)
                text_encode_b_end = time.perf_counter()

                original_size = (1024, 1024)
                crops_coords_top_left = (0, 0)
                target_size = (1024, 1024)
                add_time_ids = _get_add_time_ids(
                    unet, text_encoder2, original_size, crops_coords_top_left, target_size, dtype=encoder_hidden_states.dtype
                )
                add_time_ids = add_time_ids.to(device).repeat(args.batch_size, 1)

                noise_scheduler_b_start = time.perf_counter()
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type: {noise_scheduler.config.prediction_type}")
                noise_scheduler_b_end = time.perf_counter()

                if not args.train_text_encoder:
                    assert False
                else:
                    if enabled_dis :
                        with unet.join(), text_encoder1.join(), text_encoder2.join():
                            # Predict the noise residual and compute loss
                            noise_pred_b_start = time.perf_counter()
                            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                            with torch.autocast('cuda', enabled=args.fp16):
                                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states = encoder_hidden_states, added_cond_kwargs = added_cond_kwargs).sample
                                
                            if snr_gamma is not None :
                                # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                                # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                                # This is discussed in Section 4.2 of the same paper.
                                with torch.no_grad() :
                                    snr = compute_snr(noise_scheduler, timesteps)
                                    mse_loss_weights = (
                                        torch.stack([snr, snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                                    )
                                # We first calculate the original loss. Then we mean over the non-batch dimensions and
                                # rebalance the sample-wise losses with their respective loss weights.
                                # Finally, we take the mean of the rebalanced loss.
                                loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
                                loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights * batch_weights
                                loss = loss.mean()
                            else :
                                loss = (torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none").mean(dim=[1,2,3])*batch_weights).mean()
                            noise_pred_b_end = time.perf_counter()

                            # backprop and update
                            opt_b_start = time.perf_counter()
                            scaler.scale(loss / gas).backward()
                            if local_counter > 0 and local_counter % gas == 0 :
                                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                                torch.nn.utils.clip_grad_norm_(text_encoder1.parameters(), 1.0)
                                torch.nn.utils.clip_grad_norm_(text_encoder2.parameters(), 1.0)
                                scaler.step(optimizer)
                                scaler.update()
                                optimizer.zero_grad()
                            lr_scheduler.step()
                            opt_b_end = time.perf_counter()
                    else :
                        # Predict the noise residual and compute loss
                        noise_pred_b_start = time.perf_counter()
                        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                        with torch.autocast('cuda', enabled=args.fp16):
                            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states = encoder_hidden_states, added_cond_kwargs = added_cond_kwargs).sample
                        if torch.isnan(noise_pred).any() :
                            breakpoint()
                            
                        if snr_gamma is not None :
                            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                            # This is discussed in Section 4.2 of the same paper.
                            with torch.no_grad() :
                                snr = compute_snr(noise_scheduler, timesteps)
                                mse_loss_weights = (
                                    torch.stack([snr, snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                                )
                            # We first calculate the original loss. Then we mean over the non-batch dimensions and
                            # rebalance the sample-wise losses with their respective loss weights.
                            # Finally, we take the mean of the rebalanced loss.
                            loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
                            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights * batch_weights
                            loss = loss.mean()
                        else :
                            loss = (torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none").mean(dim=[1,2,3])*batch_weights).mean()
                        noise_pred_b_end = time.perf_counter()

                        # backprop and update
                        opt_b_start = time.perf_counter()
                        scaler.scale(loss / gas).backward()
                        if local_counter > 0 and local_counter % gas == 0 :
                            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                            torch.nn.utils.clip_grad_norm_(text_encoder1.parameters(), 1.0)
                            torch.nn.utils.clip_grad_norm_(text_encoder2.parameters(), 1.0)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                        lr_scheduler.step()
                        opt_b_end = time.perf_counter()

                # Update EMA
                ema_b_start = time.perf_counter()
                if args.use_ema:
                    if local_counter > 0 and local_counter % gas == 0 :
                        ema_unet.step(unet.parameters())
                ema_b_end = time.perf_counter()

                # perf
                b_end = time.perf_counter()
                

                reduce_b_start = time.perf_counter()
                # get global loss for logging
                if enabled_dis :
                    torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
                    loss = loss / world_size
                reduce_b_end = time.perf_counter()

                other_b_start = time.perf_counter()
                local_counter += 1

                if local_counter % 500 == 0 :
                    gc.collect()

                if rank == 0:
                    seconds_per_step = b_end - b_start
                    seconds_per_step_ema = 0.9 * seconds_per_step_ema + 0.1 * seconds_per_step
                    steps_per_second_ema = 1 / seconds_per_step_ema
                    rank_images_per_second = args.batch_size * steps_per_second_ema
                    world_images_per_second = rank_images_per_second * world_size
                    samples_seen = global_step * args.batch_size * world_size

                    progress_bar.update(1)
                    global_step += 1
                    logs = {
                        "loss": loss.detach().item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "step": global_step,
                        "samples_seen": samples_seen,
                        "rank_samples_per_second_ema": rank_images_per_second,
                        "global_samples_per_second_ema": world_images_per_second,
                        "load_overhead_ms": int(load_time_ema * 1000)
                    }
                    progress_bar.set_postfix(logs)
                    run.log(logs, step=global_step)

                    if global_step % args.save_steps == 0 and global_step > 0:
                        save_checkpoint(global_step)
                    # if global_step % 100 == 0 :
                    #     gc.collect()
                    if args.enableinference:
                        if global_step % args.image_log_steps == 0 and global_step > 0:
                            # get prompt from random batch
                            prompt = batch['prompts'][random.randint(0, len(batch['prompts'])-1)]

                            if args.image_log_scheduler == 'EulerDiscreteScheduler':
                                print('using EulerDiscreteScheduler scheduler')
                                scheduler = EulerDiscreteScheduler.from_pretrained(args.model, subfolder="scheduler", use_auth_token=args.hf_token)
                            else:
                                print('using PNDMScheduler scheduler')
                                scheduler=PNDMScheduler.from_pretrained(args.model, subfolder="scheduler", use_auth_token=args.hf_token)

                            pipeline = StableDiffusionXLPipeline(
                                text_encoder=text_encoder1 if type(text_encoder1) is not torch.nn.parallel.DistributedDataParallel else text_encoder1.module,
                                text_encoder_2=text_encoder2 if type(text_encoder2) is not torch.nn.parallel.DistributedDataParallel else text_encoder2.module,
                                vae=vae,
                                unet=unet if type(unet) is not torch.nn.parallel.DistributedDataParallel else unet.module,
                                tokenizer=tokenizer1,
                                tokenizer_2=tokenizer2,
                                scheduler=scheduler
                            ).to(device).to(torch_dtype=torch.float32)
                            # inference
                            if args.enablewandb:
                                images = []
                            else:
                                saveInferencePath = args.output_path + f"/inference-{args.run_name}"
                                os.makedirs(saveInferencePath, exist_ok=True)
                            prompt2 = 'masterpiece, best quality,1girl, solo, very long hair, brown hair, brown eyes'
                            with torch.no_grad():
                                with torch.autocast('cuda', enabled=False):
                                    for i in range(args.image_log_amount):
                                        if args.enablewandb:
                                            images.append(
                                                wandb.Image(pipeline(
                                                    prompt, num_inference_steps=args.image_log_inference_steps
                                                ).images[0],
                                                caption=prompt)
                                            )
                                        else:
                                            from datetime import datetime
                                            images = pipeline(prompt, num_inference_steps=args.image_log_inference_steps).images[0]
                                            filenameImg = str(global_step) + "-" + str(i) + ".png"
                                            filenameTxt = str(global_step) + "-" + str(i) + ".txt"
                                            images.save(saveInferencePath + "/" + filenameImg)
                                            with open(saveInferencePath + "/" + filenameTxt, 'a') as f:
                                                f.write('Used prompt: ' + prompt + '\n')
                                                f.write('Generated Image Filename: ' + filenameImg + '\n')
                                                f.write('Generated at: ' + str(global_step) + ' steps' + '\n')
                                                f.write('Generated at: ' + str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))+ '\n')

                            # log images under single caption
                            if args.enablewandb:
                                run.log({'images': images}, step=global_step)

                            # cleanup so we don't run out of memory
                            del pipeline
                            gc.collect()
                other_b_end = time.perf_counter()


                vae_b_ema = 0.9 * vae_b_ema + 0.1 * (vae_b_end - vae_b_start)
                text_encode_b_ema = 0.9 * text_encode_b_ema + 0.1 * (text_encode_b_end - text_encode_b_start)
                noise_sch_b_ema = 0.9 * noise_sch_b_ema + 0.1 * (noise_scheduler_b_end - noise_scheduler_b_start)
                noise_pred_b_ema = 0.9 * noise_pred_b_ema + 0.1 * (noise_pred_b_end - noise_pred_b_start)
                opt_b_ema = 0.9 * opt_b_ema + 0.1 * (opt_b_end - opt_b_start)
                ema_b_ema = 0.9 * ema_b_ema + 0.1 * (ema_b_end - ema_b_start)
                reduce_b_ema = 0.9 * reduce_b_ema + 0.1 * (reduce_b_end - reduce_b_start)
                other_b_ema = 0.9 * other_b_ema + 0.1 * (other_b_end - other_b_start)

                if local_counter % 200 == 0 :
                    with open(f'{args.run_name}-pref-log-{get_rank()}', 'a+') as fp :
                        fp.write('===================\n')
                        fp.write('vae_b_ema=' + str(vae_b_ema) + '\n')
                        fp.write('text_encode_b_ema=' + str(text_encode_b_ema) + '\n')
                        fp.write('noise_sch_b_ema=' + str(noise_sch_b_ema) + '\n')
                        fp.write('noise_pred_b_ema=' + str(noise_pred_b_ema) + '\n')
                        fp.write('opt_b_ema=' + str(opt_b_ema) + '\n')
                        fp.write('ema_b_ema=' + str(ema_b_ema) + '\n')
                        fp.write('reduce_b_ema=' + str(reduce_b_ema) + '\n')
                        fp.write('other_b_ema=' + str(other_b_ema) + '\n')

                start_time = time.time()
    except Exception as e:
        print(f'Exception caught on rank {rank} at step {global_step}, saving checkpoint...\n{e}\n{traceback.format_exc()}')
        pass

    save_checkpoint(global_step)

    cleanup()

    print(get_gpu_ram())
    print('Done!')


if __name__ == "__main__":
    #test_dataset()
    enabled_dis = setup()
    main(enabled_dis)