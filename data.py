from typing import Sequence, Literal
import random
import re
import cv2
import torch
import numpy as np
import cv2

import datasets
from ExperimentConfig import ExperimentConfig
from data_augmentation.data_augmentation import augment, convert_img_to_tensor
from utils import check_and_retrieveVocabulary, parse_kern
from lightning import LightningDataModule
from torch.utils.data import Dataset
from SynthGenerator import VerovioGenerator

# Single-system (music staff) preparation helpers
def prepare_data(sample, reduce_ratio=1.0, fixed_size=None):
    # HuggingFace stores PIL Images; convert to ndarray so we can resize with OpenCV.
    img = np.array(sample['image'])

    if fixed_size != None:
        width = fixed_size[1]
        height = fixed_size[0]
    elif img.shape[1] > 3056:
        width = int(np.ceil(3056 * reduce_ratio))
        height = int(np.ceil(max(img.shape[0], 256) * reduce_ratio))
    else:
        width = int(np.ceil(img.shape[1] * reduce_ratio))
        height = int(np.ceil(max(img.shape[0], 256) * reduce_ratio))

    img = cv2.resize(img, (width, height))

    gt = sample['transcription'].strip("\n ")
    gt = re.sub(r'(?<=\=)\d+', '', gt)  # Remove measure counters so the vocab stays compact.
    gt = gt.replace(" ", " <s> ")
    gt = gt.replace("·", "")
    gt = gt.replace("\t", " <t> ")
    gt = gt.replace("\n", " <b> ")

    sample["transcription"] = ["<bos>"] + gt.split(" ") + ["<eos>"]
    sample["image"] = img

    return sample

def load_set(dataset, split="train", reduce_ratio=1.0, fixed_size=None):
    # Pull a split from HuggingFace and normalize every sample with prepare_data.
    ds = datasets.load_dataset(dataset, split=split, trust_remote_code=False)
    ds = ds.map(prepare_data, fn_kwargs={"reduce_ratio": reduce_ratio, "fixed_size": fixed_size}, num_proc=4, writer_batch_size=500)

    return ds

# Full-page score preparation helpers
def prepare_fp_data(
        sample,
        reduce_ratio: float = 1.0,
        krn_format: Literal["kern"] | Literal["ekern"] | Literal["bekern"] = "bekern",
        ):
    # parse_kern rewrites **kern-like strings into whitespace-delimited tokens.
    sample["transcription"] = ['<bos>'] + parse_kern(sample["transcription"], krn_format=krn_format)[4:] + ['<eos>'] # Remove **kern, **ekern and **bekern header

    img = np.array(sample['image'].convert("RGB"))
    width = int(np.ceil(img.shape[1] * reduce_ratio))
    height = int(np.ceil(img.shape[0] * reduce_ratio))
    img = cv2.resize(img, (width, height))

    sample["image"] = img

    return sample

def load_from_files_list(
        file_ref: str,
        split: str = "train",
        krn_format: str = 'bekern',
        reduce_ratio: float = 0.5,
        map_kwargs: dict[str, any] = {"num_proc": 8}
        ):
    # Dataset entries already contain rendered images; we only need to parse tokens and resize.
    dataset = datasets.load_dataset(file_ref, split=split, trust_remote_code=False)
    # dataset.cleanup_cache_files()
    dataset = dataset.map(
            prepare_fp_data,
            fn_kwargs={
                "reduce_ratio": reduce_ratio,
                "krn_format": krn_format
                },
            **map_kwargs)

    return dataset

# Shared batching utilities
def batch_preparation_img2seq(data):
    images = [sample[0] for sample in data]
    dec_in = [sample[1] for sample in data]
    gt = [sample[2] for sample in data]

    # Determine target canvas so all images in the batch can be stacked without cropping.
    max_image_width = max(128, max([img.shape[2] for img in images]))
    max_image_height = max(256, max([img.shape[1] for img in images]))

    X_train = torch.ones(size=[len(images), 1, max_image_height, max_image_width], dtype=torch.float32)

    for i, img in enumerate(images):
        _, h, w = img.size()
        X_train[i, :, :h, :w] = img

    max_length_seq = max([len(w) for w in gt])

    # Align decoder input (no <eos>) and labels (no <bos>) to the same length.
    decoder_input = torch.zeros(size=[len(dec_in), max_length_seq-1])  # <eos> will be removed
    y = torch.zeros(size=[len(gt), max_length_seq-1])  # <bos> will be removed

    for i, seq in enumerate(dec_in):
        tokens = np.asarray([char for char in seq[:-1]])  # all tokens but <eos>
        decoder_input[i, :len(tokens)] = torch.from_numpy(tokens)

    for i, seq in enumerate(gt):
        tokens = np.asarray([char for char in seq[1:]])  # all tokens but <bos>
        y[i, :len(tokens)] = torch.from_numpy(tokens)

    return X_train, decoder_input.long(), y.long()

class OMRIMG2SEQDataset(Dataset):
    def __init__(
            self,
            teacher_forcing_error_rate: float = .2,
            augment: bool = False,
            *args, **kwargs
            ) -> None:
        super().__init__(*args, **kwargs)

        self.teacher_forcing_error_rate = teacher_forcing_error_rate
        self.augment = augment

        self.x: Sequence
        self.y: Sequence

    def apply_teacher_forcing(self, sequence):
        # Randomly corrupt tokens so the decoder learns to recover from mistakes during training.
        errored_sequence = sequence.clone()
        for token in range(1, len(sequence)):
            if np.random.rand() < self.teacher_forcing_error_rate and sequence[token] != self.padding_token:
                errored_sequence[token] = np.random.randint(0, len(self.w2i))

        return errored_sequence

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        if self.augment:
            x = augment(np.array(self.x[index]))
        else:
            x = convert_img_to_tensor(np.array(self.x[index]))

        y = torch.from_numpy(np.asarray([self.w2i[token] for token in self.y[index]]))
        decoder_input = self.apply_teacher_forcing(y)

        return x, decoder_input, y

    def get_max_hw(self):
        m_height = np.max([img.size[1] for img in self.x])
        m_width = np.max([img.size[0] for img in self.x])

        return m_height, m_width

    def get_max_seqlen(self):
        return np.max([len(seq) for seq in self.y])

    def vocab_size(self):
        return len(self.w2i)

    def get_gt(self):
        return self.y

    def set_dictionaries(self, w2i, i2w):
        self.w2i = w2i
        self.i2w = i2w
        self.padding_token = w2i['<pad>']

    def get_dictionaries(self):
        return self.w2i, self.i2w

    def get_i2w(self):
        return self.i2w

class GrandStaffSingleSystem(OMRIMG2SEQDataset):
    """System-level dataset from huggingface"""

    def __init__(
            self,
            data_path,
            split,
            teacher_forcing_error_rate: float = .2,
            augment: bool = False,
            reduce_ratio: float = 1.0,
            *args, **kwargs
            ) -> None:
        super().__init__(
                teacher_forcing_error_rate=teacher_forcing_error_rate,
                augment=augment,
                *args, **kwargs
                )
        self.reduce_ratio: float = reduce_ratio

        # Lazily pull the desired split from HuggingFace; preprocessing happens in prepare_data.
        self.data = load_set(data_path, split, reduce_ratio=reduce_ratio)

    def get_width_avgs(self):
        widths = [s["image"].size[0] for s in self.data]

        return np.average(widths), np.max(widths), np.min(widths)

    def get_max_height(self) -> int:
        return np.max([s["image"].size[1] for s in self.data])

    def get_max_width(self) -> int:
        return np.max([s["image"].size[0] for s in self.data])

    def get_max_seqlen(self):
        return np.max([len(s["transcription"]) for s in self.data])

    def get_max_hw(self):
        m_height = np.max([s["image"].size[1] for s in self.data])
        m_width = np.max([s["image"].size[0] for s in self.data])

        return m_height, m_width

    def __getitem__(self, index):
        sample = self.data[index]

        x = np.array(sample["image"])
        y = sample["transcription"]

        if self.augment:
            x = augment(x)
        else:
            x = convert_img_to_tensor(x)

        y = torch.from_numpy(np.asarray([self.w2i[token] for token in y]))
        decoder_input = self.apply_teacher_forcing(y)

        return x, decoder_input, y

    def __len__(self):
        return len(self.data)

    def get_gt(self):
        return self.data["transcription"]

class GrandStaffFullPage(GrandStaffSingleSystem):
    """Full-page dataset from huggingface"""

    def __init__(
            self,
            data_path: str,
            split: str = "train",
            teacher_forcing_error_rate: float = 0.2,
            reduce_ratio: float = 1.0,
            augment: bool = False,
            krn_format: str = "bekern",
            *args, **kwargs
            ):
        OMRIMG2SEQDataset.__init__(
                self,
                teacher_forcing_error_rate=teacher_forcing_error_rate,
                augment=augment,
                *args, **kwargs
                )
        self.reduce_ratio: float = reduce_ratio
        self.krn_format: str = krn_format

        # Full pages need the kern parser (headers removed) rather than the lightweight single-system path.
        self.data = load_from_files_list(data_path, split, krn_format, reduce_ratio=reduce_ratio, map_kwargs={"writer_batch_size": 32})

class SyntheticOMRDataset(OMRIMG2SEQDataset):
    """Synthetic dataset using VerovioGenerator"""
    def __init__(
            self,
            data_path: str,
            split: str = "train",
            number_of_systems: int = 1,
            teacher_forcing_error_rate: float = 0.2,
            reduce_ratio: float = .5,
            dataset_length: int = 40000,
            augment: bool = False,
            krn_format: str = "bekern"
            ) -> None:
        super().__init__(teacher_forcing_error_rate, augment)
        # VerovioGenerator renders pseudo-random scores on the fly so we never store synthetic images on disk.
        self.generator = VerovioGenerator(sources=data_path, split=split, krn_format=krn_format)

        self.num_sys_gen: int = number_of_systems
        self.dataset_len: int = dataset_length
        self.reduce_ratio: float = reduce_ratio
        self.krn_format: str = krn_format

        self.x = None
        self.y = None

    def __getitem__(self, index):
        x, y = self.generator.generate_music_system_image()

        x = np.array(x)

        if self.augment:
            x = augment(x)
        else:
            x = convert_img_to_tensor(x)

        y = torch.from_numpy(np.asarray([self.w2i[token] for token in y]))
        decoder_input = self.apply_teacher_forcing(y)

        return x, decoder_input, y

    def __len__(self):
        return self.dataset_len

# NOTE: Synthetic GrandStaff system-level for pre-training
# NOTE: GrandStaff Curriculum Learning for system-to-page curriculum training
class CurriculumTrainingDataset(GrandStaffFullPage):
    def __init__(
            self,
            data_path: str,
            synthetic_sources: str,
            split: str = "train",
            teacher_forcing_error_rate: float = 0.2,
            reduce_ratio: float = 1.0,
            augment: bool = False,
            krn_format: str = "bekern",
            skip_steps: int = 0,
            *args, **kwargs
            ) -> None:
        self.generator = VerovioGenerator(sources=synthetic_sources, split=split, krn_format=krn_format)
        super().__init__(
                data_path=data_path,
                split=split,
                teacher_forcing_error_rate=teacher_forcing_error_rate,
                reduce_ratio=reduce_ratio,
                augment=augment,
                krn_format=krn_format,
                *args, **kwargs
                )

        self.max_synth_prob: float = 0.9
        self.min_synth_prob: float = 0.2
        self.finetune_steps: int = 200000
        self.increase_steps: int = 40000
        self.skip_cl_steps: int = skip_steps
        self.num_cl_steps: int = 3
        self.max_cl_steps: int = self.increase_steps * self.num_cl_steps
        self.num_cl_steps -= self.skip_cl_steps
        self.curriculum_stage_beginning: int = 2 + self.skip_cl_steps

    def set_trainer_data(self, trainer):
        # Lightning trainer injected after datamodule setup so curriculum can inspect global_step.
        self.trainer = trainer

    def linear_scheduler_synthetic(self, step):
        step += self.skip_cl_steps * self.increase_steps
        # After curriculum phases, gradually bias sampling toward real pages for fine-tuning.
        return self.max_synth_prob + round((step - self.max_cl_steps) * (self.min_synth_prob - self.max_synth_prob) / self.finetune_steps, 4)

    def get_stage_calculator(self):
        step_increase: int = self.increase_steps
        curriculum_start: int = self.curriculum_stage_beginning

        def stage_calculator(step):
            return (step // step_increase) + curriculum_start

        return stage_calculator

    def __getitem__(self, index):
        step = self.trainer.global_step
        stage = (step // self.increase_steps) + self.curriculum_stage_beginning

        gen_author_title = np.random.rand() > 0.5

        if stage < (self.num_cl_steps + self.curriculum_stage_beginning):
           # Early curriculum: constrain synthetic pages to the current stage's system count.
           num_sys_to_gen = random.randint(1, stage)
           x, y = self.generator.generate_full_page_score(
               max_systems = num_sys_to_gen,
               strict_systems=True,
               strict_height=(random.random() < 0.3),
               include_author=gen_author_title,
               include_title=gen_author_title,
               reduce_ratio=self.reduce_ratio)
        else:
            probability = max(self.linear_scheduler_synthetic(step), self.min_synth_prob)
            # wandb.log({'Synthetic Probability': probability}, commit=False)

            if random.random() > probability:
                # Sample a real score to finish fine-tuning on genuine layouts.
                x = self.data[index]["image"]
                y = self.data[index]["transcription"]
            else:
                # Otherwise keep generating richer synthetic pages as regularization.
                x, y = self.generator.generate_full_page_score(
                    max_systems = random.randint(3, 4),
                    strict_systems=False,
                    strict_height=(random.random() < 0.3),
                    include_author=gen_author_title,
                    include_title=gen_author_title,
                    reduce_ratio=self.reduce_ratio)

        x = np.array(x)

        if self.augment:
           x = augment(x)
        else:
           x = convert_img_to_tensor(x)

        y = torch.from_numpy(np.asarray([self.w2i[token] for token in y if token != '']))
        decoder_input = self.apply_teacher_forcing(y)

        # wandb.log({'Stage': stage})

        return x, decoder_input, y

class GrandStaffFullPageCurriculumLearning(CurriculumTrainingDataset):
    def __init__(
            self,
            data_path: str,
            synthetic_sources: str = "antoniorv6/grandstaff-ekern",
            split: str = "train",
            teacher_forcing_error_rate: float = 0.2,
            reduce_ratio: float = .5,
            augment: bool = False,
            krn_format: str = "bekern",
            skip_steps: int = 0,
            *args, **kwargs
            ) -> None:
       super().__init__(
                data_path=data_path,
                synthetic_sources=synthetic_sources,
                split=split,
                teacher_forcing_error_rate=teacher_forcing_error_rate,
                reduce_ratio=reduce_ratio,
                augment=augment,
                krn_format=krn_format,
                skip_steps=skip_steps,
                *args, **kwargs
                )

# Huggingface system-level GrandStaff training
class GrandStaffDataset(LightningDataModule):
    def __init__(self, config:ExperimentConfig) -> None:
        super().__init__()
        self.data_path = config.data_path
        self.vocab_name = config.vocab_name
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.train_set = GrandStaffSingleSystem(data_path=self.data_path, split="train", augment=True)
        self.val_set = GrandStaffSingleSystem(data_path=self.data_path, split="val",)
        self.test_set = GrandStaffSingleSystem(data_path=self.data_path, split="test",)

        # Build a consistent vocab across splits so token ids align during training/validation/testing.
        w2i, i2w = check_and_retrieveVocabulary([self.train_set.get_gt(), self.val_set.get_gt(), self.test_set.get_gt()], "vocab/", f"{self.vocab_name}")

        self.train_set.set_dictionaries(w2i, i2w)
        self.val_set.set_dictionaries(w2i, i2w)
        self.test_set.set_dictionaries(w2i, i2w)

    def get_max_height(self) -> int:
        Th = self.train_set.get_max_height()
        vh = self.val_set.get_max_height()
        th = self.test_set.get_max_height()

        return max(Th, vh, th)

    def get_max_width(self) -> int:
        Tw = self.train_set.get_max_width()
        vw = self.val_set.get_max_width()
        tw = self.test_set.get_max_width()

        return max(Tw, vw, tw)

    def get_max_length(self) -> int:
        Tl = self.train_set.get_max_seqlen()
        vl = self.val_set.get_max_seqlen()
        tl = self.test_set.get_max_seqlen()

        return max(Tl, vl, tl)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=batch_preparation_img2seq)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=batch_preparation_img2seq)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=batch_preparation_img2seq)

# Synthetic system-level GrandStaff training
# NOTE: Pre-train the SMT on system-level data using this dataset
class SyntheticGrandStaffDataset(LightningDataModule):
    def __init__(self, config:ExperimentConfig) -> None:
        super().__init__()
        self.data_path = config.data_path
        self.vocab_name = config.vocab_name
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.krn_format = config.krn_format

        self.train_set: SyntheticOMRDataset = SyntheticOMRDataset(data_path=self.data_path, split="train", dataset_length=40000, augment=True, krn_format=self.krn_format)
        self.val_set: SyntheticOMRDataset = SyntheticOMRDataset(data_path=self.data_path, split="val", dataset_length=1000, augment=False, krn_format=self.krn_format)
        self.test_set: SyntheticOMRDataset = SyntheticOMRDataset(data_path=self.data_path, split="test", dataset_length=1000, augment=False, krn_format=self.krn_format)
        # Synthetic samples still rely on a persisted vocab so downstream checkpoints remain compatible.
        w2i, i2w = check_and_retrieveVocabulary([self.train_set.get_gt(), self.val_set.get_gt(), self.test_set.get_gt()], "vocab/", f"{self.vocab_name}")#

        self.train_set.set_dictionaries(w2i, i2w)
        self.val_set.set_dictionaries(w2i, i2w)
        self.test_set.set_dictionaries(w2i, i2w)

    def get_max_height(self) -> int:
        return 2512

    def get_max_width(self) -> int:
        return 2512

    def get_max_length(self) -> int:
        return 4360

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=batch_preparation_img2seq)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=batch_preparation_img2seq)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=batch_preparation_img2seq)

# Synthetic system-to-full-page GrandStaff curriculum training
# NOTE: Fine-tune the SMT on page-level data with curriculum learning
# NOTE: using this dataset
class SyntheticCLGrandStaffDataset(LightningDataModule):
    def __init__(self, config:ExperimentConfig, skip_steps: int = 0, fold: int = 0) -> None:
        super().__init__()
        self.data_path = config.data_path
        self.vocab_name = config.vocab_name
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.krn_format = config.krn_format

        self.train_set = GrandStaffFullPageCurriculumLearning(data_path=self.data_path, split="train", augment=True, krn_format=self.krn_format, reduce_ratio=config.reduce_ratio, skip_steps=skip_steps)
        self.val_set = GrandStaffFullPage(data_path=self.data_path, split="val", augment=False, krn_format=self.krn_format, reduce_ratio=config.reduce_ratio)
        self.test_set = GrandStaffFullPage(data_path=self.data_path, split="test", augment=False, krn_format=self.krn_format, reduce_ratio=config.reduce_ratio)
        # Ensure curriculum stages and evaluation share token ids; vocab is derived jointly across splits.
        w2i, i2w = check_and_retrieveVocabulary([self.train_set.get_gt(), self.val_set.get_gt(), self.test_set.get_gt()], "vocab/", f"{self.vocab_name}")#

        self.train_set.set_dictionaries(w2i, i2w)
        self.val_set.set_dictionaries(w2i, i2w)
        self.test_set.set_dictionaries(w2i, i2w)

    def get_max_height(self) -> int:
        Th = self.train_set.get_max_height()
        vh = self.val_set.get_max_height()
        th = self.test_set.get_max_height()

        return max(Th, vh, th, 2970)

    def get_max_width(self) -> int:
        Tw = self.train_set.get_max_width()
        vw = self.val_set.get_max_width()
        tw = self.test_set.get_max_width()

        return max(Tw, vw, tw, 2100)

    def get_max_length(self) -> int:
        Tl = self.train_set.get_max_seqlen()
        vl = self.val_set.get_max_seqlen()
        tl = self.test_set.get_max_seqlen()

        return max(Tl, vl, tl, 4353)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=batch_preparation_img2seq)
        # return torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, num_workers=0, shuffle=True, collate_fn=batch_preparation_img2seq)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=batch_preparation_img2seq)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=batch_preparation_img2seq)
