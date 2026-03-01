from functools import partial
from typing import Dict

from .raw_data import (
    RawDatasetPreprocessor,
    HhRlhfRDP,
    PKUSafeRlhfRDP, PKUSafeRlhf10KRDP,
    SHPRDP,
    StackExchangePairedRDP,
    SummarizeFromFeedbackRDP, 
    HelpSteerRDP,
    UltraFeedbackRDP,
)
from .raw_data.utils import DEFAULT_PROMPT_TEMPLATE


REAL_DATASET_CONFIGS: Dict[str, RawDatasetPreprocessor] = {
    ##### hh-rlhf (https://huggingface.co/datasets/Anthropic/hh-rlhf) #####
    "Anthropic/hh-rlhf": HhRlhfRDP,
    "Anthropic/hh-rlhf-helpful": partial(HhRlhfRDP, subset="helpful-base"),
    "Anthropic/hh-rlhf-harmless": partial(HhRlhfRDP, subset="harmless-base"),

    ##### PKU-SafeRLHF (https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF) #####
    **{
        f"PKU-Alignment/PKU-SafeRLHF-{dimension}": partial(PKUSafeRlhfRDP, dimension=dimension)
        for dimension in ["safer", "better"]
    },
    **{
        f"PKU-Alignment/PKU-SafeRLHF-10K-{dimension}": partial(PKUSafeRlhf10KRDP, dimension=dimension)
        for dimension in ["safer", "better"]
    },

    ##### stack-exchange-paired (https://huggingface.co/datasets/lvwerra/stack-exchange-paired) #####
    "lvwerra/stack-exchange-paired": StackExchangePairedRDP,

    ##### SHP (https://huggingface.co/datasets/stanfordnlp/SHP) #####
    "stanfordnlp/SHP": SHPRDP,

    ##### summarize_from_feedback (https://huggingface.co/datasets/openai/summarize_from_feedback) #####
    "openai/summarize_from_feedback": SummarizeFromFeedbackRDP,

    ##### UltraFeedback (https://huggingface.co/datasets/openbmb/UltraFeedback) #####
    "OpenBMB/UltraFeedback": UltraFeedbackRDP,
    **{
        f"OpenBMB/UltraFeedback-{dimension}": partial(UltraFeedbackRDP, dimension=dimension)
        for dimension in ["overall", "instruction_following", "honesty", "truthfulness", "helpfulness"]
    },
    # Curated UF variants for stricter pair construction.
    "OpenBMB/UltraFeedback-helpfulness-clean-gap2": partial(
        UltraFeedbackRDP,
        dimension="helpfulness",
        min_gap=2.0,
    ),
    "OpenBMB/UltraFeedback-truthfulness-clean-gap2": partial(
        UltraFeedbackRDP,
        dimension="truthfulness",
        min_gap=2.0,
    ),
    "OpenBMB/UltraFeedback-truthfulness-clean-gap2-joint-with-helpfulness-gap2": partial(
        UltraFeedbackRDP,
        dimension="truthfulness",
        min_gap=2.0,
        anchor_dimension="helpfulness",
        min_anchor_gap=2.0,
    ),
    "OpenBMB/UltraFeedback-helpfulness-clean-gap0": partial(
        UltraFeedbackRDP,
        dimension="helpfulness",
        min_gap=0.0,
    ),
    "OpenBMB/UltraFeedback-truthfulness-clean-gap0": partial(
        UltraFeedbackRDP,
        dimension="truthfulness",
        min_gap=0.0,
    ),
    "OpenBMB/UltraFeedback-truthfulness-clean-gap0-joint-with-helpfulness-gap0": partial(
        UltraFeedbackRDP,
        dimension="truthfulness",
        min_gap=0.0,
        anchor_dimension="helpfulness",
        min_anchor_gap=0.0,
    ),
    "OpenBMB/UltraFeedback-truthfulness-clean-gap2-conflict-with-helpfulness-gap2": partial(
        UltraFeedbackRDP,
        dimension="truthfulness",
        min_gap=2.0,
        anchor_dimension="helpfulness",
        min_anchor_gap=2.0,
        require_disagreement_with_anchor=True,
    ),
    "OpenBMB/UltraFeedback-truthfulness-clean-gap2-agreement-with-helpfulness-gap2": partial(
        UltraFeedbackRDP,
        dimension="truthfulness",
        min_gap=2.0,
        anchor_dimension="helpfulness",
        min_anchor_gap=2.0,
        require_agreement_with_anchor=True,
    ),
    "OpenBMB/UltraFeedback-truthfulness-clean-gap0-conflict-with-helpfulness-gap0": partial(
        UltraFeedbackRDP,
        dimension="truthfulness",
        min_gap=0.0,
        anchor_dimension="helpfulness",
        min_anchor_gap=0.0,
        require_disagreement_with_anchor=True,
    ),
    "OpenBMB/UltraFeedback-truthfulness-clean-gap0-agreement-with-helpfulness-gap0": partial(
        UltraFeedbackRDP,
        dimension="truthfulness",
        min_gap=0.0,
        anchor_dimension="helpfulness",
        min_anchor_gap=0.0,
        require_agreement_with_anchor=True,
    ),

    ##### HelpSteer (https://huggingface.co/datasets/nvidia/HelpSteer) #####
    "nvidia/HelpSteer": HelpSteerRDP,
    **{
        f"nvidia/HelpSteer-pairwise-{dimension}": partial(HelpSteerRDP, dimension=dimension)
        for dimension in ["overall", "helpfulness", "correctness", "coherence", "complexity", "verbosity"]
    },
}


# !WARNING: Synthetic datasets are WIP. These configs are just placeholders 
SYNTHETIC_DATASET_CONFIGS = {

}


# MERGE two dicts
DATASET_CONFIGS = {**REAL_DATASET_CONFIGS, **SYNTHETIC_DATASET_CONFIGS}
