"""!
@package jev_27b
@brief JEV-27B-VL (autotrust/JEV-27B-VL): the same questions in its decision format (`choice` kind, no chat template),
       the rungs as options A-G; its base model with its LoRA decision adapter, the rung read from the option letters
       plus the decision head's bias and calibration temperature, as its model card does without its server. It has
       no option for cannot_judge.
"""
import json
from pathlib import Path

from inference.vlm.base import VisionLanguageModel, prompt_instructions, term_definitions


class JEV27BVL(VisionLanguageModel):

    def build_prompt(self, field):
        """!
        @brief The question for one term in JEV's decision format: the rungs become options A, B, C ...; the image
               sits in [state].
        """
        options = "\n".join(f"{'ABCDEFG'[i]}) {r}" for i, r in enumerate(term_definitions(self.V)[field]["scale"]))
        return ("[kind] choice\n[state] <|vision_start|><|image_pad|><|vision_end|>\n[question] "
                + prompt_instructions(field, self.V["prompts"]["choose_option"], self.V)
                + f"\n[options]\n{options}\n[decision]:")

    def load_model(self):
        """!
        @brief The base model with JEV's decision adapter, its decision head and calibration.
        """
        from huggingface_hub import hf_hub_download
        from peft import PeftModel
        name = self.V["model"]
        self.proc, model = self.load_processor_and_model()
        self.model = PeftModel.from_pretrained(model, name, subfolder="adapter_vllm").eval()
        head = json.loads(Path(hf_hub_download(name, "adapter_vllm/decision_head.json")).read_text())
        self.temperature = json.loads(Path(hf_hub_download(name, "calibration.json")).read_text())["per_kind"]["choice"]
        first = head["slots"]["ranges"]["choice"][0]                # option letters A, B, C ...
        self.texts = {f: self.build_prompt(f) for f in term_definitions(self.V)}
        self.ids, self.bias = head["verbalizer_ids"][first:first + 7], head["bias"][first:first + 7]
