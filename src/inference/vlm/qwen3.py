"""!
@package qwen3
@brief Qwen-VL chat models (Qwen/Qwen3-VL-32B-Instruct): each term asked in a chat prompt whose answer is forced to
       begin {"<term>": , so the next token is the rung, read from the digit tokens 1-7 (and 0, cannot judge, for the
       terms that have it).
"""
import sys

from inference.vlm.base import RUNGS, VisionLanguageModel, prompt_instructions, term_definitions


class Qwen3VL(VisionLanguageModel):

    @property
    def system(self):
        return self.V["prompts"]["system"]

    def build_prompt(self, field):
        """!
        @brief The user prompt for one term: the view, the term's definition, what to look at, its rungs (with rung 0
               if the term has cannot_judge), the reply format.
        """
        P, c = self.V["prompts"], term_definitions(self.V)[field]
        rungs = [f"{i + 1} = {r}" for i, r in enumerate(c["scale"])]
        if "cannot_judge" in c:
            rungs.insert(0, f"0 = cannot judge: {c['cannot_judge']}")
        return (prompt_instructions(field, P["choose_rung"], self.V) + "\n\n"
                + P["scale_and_reply"].format(scale="\n".join(rungs), field=field,
                                              lowest=0 if "cannot_judge" in c else 1))

    def load_model(self):
        """!
        @brief The model (bf16) and its prompts as chat texts ending in the forced answer prefix.
        """
        V = self.V
        self.proc, model = self.load_processor_and_model()
        self.model = model.eval()
        self.texts = {f: self.proc.apply_chat_template(
            [{"role": "system", "content": self.system},
             {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.build_prompt(f)}]}],
            tokenize=False, add_generation_prompt=True) + f'{{"{f}": ' for f in term_definitions(V)}
        enc = [self.proc.tokenizer.encode(str(k), add_special_tokens=False) for k in (0, *RUNGS)]
        if any(len(i) != 1 for i in enc):
            sys.exit(f"{V['model']}: the digits 0-7 are not single tokens, so the rung distribution cannot be read")
        self.ids, self.zero = [i[0] for i in enc[1:]], enc[0][0]
        self.cannot_judge = {f for f, c in term_definitions(V).items() if "cannot_judge" in c}
