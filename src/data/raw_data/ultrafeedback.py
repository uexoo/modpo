from dataclasses import dataclass
from functools import partial
from typing import Dict, Literal, Optional

from datasets import load_dataset

from .utils import RawDatasetPreprocessor
from src.utils import print_local_main

UF_FINE_DIMS = ("instruction_following", "honesty", "truthfulness", "helpfulness")
UF_DIMS = UF_FINE_DIMS + ("overall",)


def _to_float_or_none(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _uf_score_for_dimension(completion: dict, dimension: str) -> Optional[float]:
    if dimension == "overall":
        return _to_float_or_none(completion.get("overall_score"))
    ann = completion.get("annotations", {})
    dim = ann.get(dimension, {}) if isinstance(ann, dict) else {}
    rating = dim.get("Rating") if isinstance(dim, dict) else None
    return _to_float_or_none(rating)


def _chosen_id_and_gap(score_0: Optional[float], score_1: Optional[float]) -> tuple[int, float]:
    if score_0 is None or score_1 is None:
        return -1, 0.0
    diff = score_0 - score_1
    if diff > 0:
        return 0, abs(diff)
    if diff < 0:
        return 1, abs(diff)
    return -1, 0.0


def ultrafeedback_transform_to_sft(batched_sample, prompt_template):
    new_batched_sample = {
        "raw_prompt": [],
        "prompt": [],
        "response": [],
    }
    for instruction, completions in zip(batched_sample["instruction"], batched_sample["completions"]):
        # breakpoint()
        for completion in completions:
            new_batched_sample["raw_prompt"].append(instruction)
            new_batched_sample["prompt"].append(prompt_template.format(raw_prompt=instruction))
            new_batched_sample["response"].append(completion["response"])
    return new_batched_sample


def ultrafeedback_transform_to_preference(batched_sample):
    new_batched_sample = {
        "prompt": [],
        "response_0": [],
        "response_1": [],
        **{f"{dimension}_chosen_id": [] for dimension in UF_DIMS},
        **{f"{dimension}_gap": [] for dimension in UF_DIMS},
    }
    for instruction, completions in zip(batched_sample["instruction"], batched_sample["completions"]):
        n_responses = len(completions)

        for j in range(n_responses):
            for k in range(j + 1, n_responses):
                new_batched_sample["prompt"].append(instruction)
                new_batched_sample["response_0"].append(completions[j]["response"])
                new_batched_sample["response_1"].append(completions[k]["response"])
                for dimension in UF_DIMS:
                    score_0 = _uf_score_for_dimension(completions[j], dimension)
                    score_1 = _uf_score_for_dimension(completions[k], dimension)
                    cid, gap = _chosen_id_and_gap(score_0, score_1)
                    new_batched_sample[f"{dimension}_chosen_id"].append(cid)
                    new_batched_sample[f"{dimension}_gap"].append(gap)

    return new_batched_sample


@dataclass
class UltraFeedbackRDP(RawDatasetPreprocessor):
    path: Optional[str] = "OpenBMB/UltraFeedback"
    dimension: Optional[Literal["overall", "instruction_following", "honesty", "truthfulness", "helpfulness"]] = None
    min_gap: float = 1.0
    anchor_dimension: Optional[
        Literal["overall", "instruction_following", "honesty", "truthfulness", "helpfulness"]
    ] = None
    min_anchor_gap: float = 1.0
    require_disagreement_with_anchor: bool = False

    def _get_raw_dataset(self, split):
        if split == "train":
            return load_dataset(self.path, split="train").train_test_split(test_size=0.1, seed=0)["train"]
        elif split == "validation":
            return load_dataset(self.path, split="train").train_test_split(test_size=0.1, seed=0)["test"]
        elif split == "test":
            raise NotImplementedError("test split not implemented for UltraFeedbackRDP")
        else:
            raise NotImplementedError

    def _dataset_to_preference_formatter(self, example) -> Dict[str, str]:
        chosen_id = example[f"{self.dimension}_chosen_id"]
        return {
            "raw_prompt": example["prompt"],
            "prompt": self.prompt_template.format(raw_prompt=example["prompt"]),
            "chosen": example[f"response_{chosen_id}"],
            "rejected": example[f"response_{1-chosen_id}"],
        }

    def get_preference_dataset(self, split):
        if not self.dimension:
            raise ValueError("UltraFeedback preference requires --dimension.")
        if self.min_gap <= 0:
            raise ValueError(f"min_gap must be > 0. Got: {self.min_gap}")
        if self.anchor_dimension and self.min_anchor_gap <= 0:
            raise ValueError(f"min_anchor_gap must be > 0. Got: {self.min_anchor_gap}")

        dataset = self._get_raw_dataset(split)
        if self.sanity_check:
            dataset = dataset.select(range(min(len(dataset), 100)))
        print_local_main("mapping raw dataset to preference...")
        dataset = dataset.map(
            ultrafeedback_transform_to_preference,
            batched=True,
            num_proc=self.num_proc,
            remove_columns=dataset.column_names,
        )
        print_local_main("filtering preference...")
        dataset = dataset.filter(
            lambda x: x[f"{self.dimension}_chosen_id"] != -1 and x[f"{self.dimension}_gap"] >= float(self.min_gap)
        )
        if self.anchor_dimension:
            dataset = dataset.filter(
                lambda x: x[f"{self.anchor_dimension}_chosen_id"] != -1
                and x[f"{self.anchor_dimension}_gap"] >= float(self.min_anchor_gap)
            )
            if self.require_disagreement_with_anchor:
                dataset = dataset.filter(
                    lambda x: x[f"{self.dimension}_chosen_id"] != x[f"{self.anchor_dimension}_chosen_id"]
                )
        print_local_main("mapping dataset to standard format...")
        return dataset.map(
            self._dataset_to_preference_formatter,
            num_proc=self.num_proc,
            remove_columns=dataset.column_names,
        )

    def get_sft_dataset(self, split, **kwargs):
        if self.dimension:
            return super().get_sft_dataset(split, **kwargs)
        dataset = self._get_raw_dataset(split)
        if self.sanity_check:
            dataset = dataset.select(range(min(len(dataset), 100)))
        print_local_main("mapping raw dataset to sft...")
        return dataset.map(
            partial(ultrafeedback_transform_to_sft, prompt_template=self.prompt_template),
            batched=True,
            num_proc=self.num_proc,
            remove_columns=dataset.column_names,
        )


if __name__ == "__main__":
    num_proc = 4
    overall_dataset = UltraFeedbackRDP(dimension="overall", num_proc=num_proc).get_preference_dataset(split="train")
    sft_dataset     = UltraFeedbackRDP(num_proc=num_proc).get_sft_dataset(split="train")
    breakpoint()
