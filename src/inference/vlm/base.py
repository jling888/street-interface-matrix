"""!
@package base
@brief The base of every VLM in vlm/: what every model is asked (the terms, their view, the words every prompt
       shares) and how a rating is read from it (rung_probabilities: the rung distribution from the logits).

A model is a sibling file with one subclass: its prompt format (build_prompt) and its loading (load_model). run_vlm.py
creates the one vlm.model_file names and does everything else: the views, combining a facade's sides, the
table.
"""
import hashlib
import sys
from abc import ABC, abstractmethod

import numpy as np

RUNGS = np.arange(1, 8)                                                  # the rungs


def term_definitions(V):
    """!
    @brief The terms rated (configs/inference/vlm_prompts.yaml, in the config's `vlm`).
    """
    return V["prompts"]["terms"]


def is_facade_term(field, V):
    """!
    @brief Whether `field` is rated on the two frames square to the frontage.
    """
    return term_definitions(V)[field].get("view") == "facade"


def prompt_instructions(field, choose, V):
    """!
    @brief What every prompt for a term says before its scale: the view, the definition, what to look at.
    @param choose  The last words (the prompts' choose_rung or choose_option).
    """
    P, c = V["prompts"], term_definitions(V)[field]
    view = P["view_sentences"]["facade" if is_facade_term(field, V) else "along"]
    return P["instructions"].format(view=view, definition=c["definition"], check=c["check"], choose=choose)


class VisionLanguageModel(ABC):
    """!
    @brief A VLM rating the terms. load_model() sets proc, model, texts ({term: its input text}) and ids (the token of
           each rung 1-7); the class attributes below hold unless a model sets its own.
    """
    system = ""                    # the system prompt, if the model's format has one
    bias = (0.0,) * 7              # added to each rung's logit
    temperature = 1.0              # the rung logits are divided by it
    zero = None                    # rung 0's token, for the terms with cannot_judge
    cannot_judge = frozenset()     # the terms with a rung 0

    def __init__(self, V):
        """!
        @param V  The config's `vlm`.
        """
        self.V = V

    @abstractmethod
    def build_prompt(self, field):
        """!
        @brief The prompt for one term (a key of term_definitions(V)).
        """

    @abstractmethod
    def load_model(self):
        """!
        @brief Load the weights and the prompts as the model's input texts.
        """

    def prompts_fingerprint(self):
        """!
        @brief A short fingerprint of the prompts a table is rated with: one rated with other prompts must not be
               resumed with these.
        """
        text = self.system + "".join(self.build_prompt(f) for f in term_definitions(self.V))
        return hashlib.sha1(text.encode()).hexdigest()[:10]

    def load_processor_and_model(self):
        """!
        @brief The model's processor (padding on the left: batched rows must end at the same position) and its
               weights in bf16 on the GPU, for load_model().
        """
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        if not torch.cuda.is_available():
            sys.exit("rating with the VLM needs a CUDA GPU: src/colab.sh runs this step on Colab")
        name = self.V["model"]
        proc = AutoProcessor.from_pretrained(name)
        proc.tokenizer.padding_side = "left"
        return proc, AutoModelForImageTextToText.from_pretrained(name, device_map="cuda", dtype=torch.bfloat16)

    def rung_probabilities(self, images):
        """!
        @brief One view's rung distribution for each term asked, the terms on one image in one batch.
        @param images  {key: the image it is rated on}, a key a field or (field, side) (run_vlm.view_images()).
        @return ({key: probabilities of rungs 1..7}, renormalised over the rung tokens after bias and temperature;
                {key: the probability of rung 0, against all eight}, for the terms with cannot_judge).
        """
        import torch
        out, cannot = {}, {}
        for im in {id(i): i for i in images.values()}.values():
            fs = [f for f, i in images.items() if i is im]
            inp = self.proc(text=[self.texts[f[0] if isinstance(f, tuple) else f] for f in fs], images=[im] * len(fs),
                            padding=True, return_tensors="pt").to("cuda")
            with torch.no_grad():
                last = self.model(**inp).logits[:, -1].float()
            logits = (last[:, self.ids] + torch.tensor(self.bias, device=last.device)) / self.temperature
            out.update(zip(fs, torch.softmax(logits, -1).cpu().numpy()))
            for i, f in enumerate(fs):
                if (f[0] if isinstance(f, tuple) else f) in self.cannot_judge:
                    cannot[f] = float(torch.softmax(torch.cat([last[i, [self.zero]], logits[i]]), -1)[0])
        return {f: out[f] for f in images}, cannot
