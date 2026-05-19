"""Prompt templates for various chat formats."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict

TEMPLATES: Dict[str, "PromptTemplate"] = {}


@dataclass
class PromptTemplate:
    name: str
    system_prefix: str = ""
    system_suffix: str = ""
    human_prefix: str = "Human: "
    human_suffix: str = "\n"
    assistant_prefix: str = "Assistant: "
    assistant_suffix: str = "\n"
    stop_words: List[str] = None

    def __post_init__(self):
        TEMPLATES[self.name] = self
        if self.stop_words is None:
            self.stop_words = []

    def format_example(self, instruction: str, input_text: str = "", output: str = "") -> str:
        prompt = ""
        user_content = instruction
        if input_text:
            user_content = f"{instruction}\n\n{input_text}"
        prompt += self.human_prefix + user_content + self.human_suffix
        prompt += self.assistant_prefix + output + self.assistant_suffix
        return prompt

    def format_messages(self, messages: List[Dict]) -> str:
        result = ""
        system = next((m["value"] for m in messages if m.get("from") == "system"), "")
        if system:
            result += self.system_prefix + system + self.system_suffix
        for msg in messages:
            role = msg.get("from", "")
            value = msg.get("value", "")
            if role in ("human", "user"):
                result += self.human_prefix + value + self.human_suffix
            elif role in ("gpt", "assistant"):
                result += self.assistant_prefix + value + self.assistant_suffix
        return result


PromptTemplate(
    name="alpaca",
    human_prefix="### Instruction:\n",
    human_suffix="\n\n",
    assistant_prefix="### Response:\n",
    assistant_suffix="\n\n",
)

PromptTemplate(
    name="chatml",
    human_prefix="<|im_start|>user\n",
    human_suffix="<|im_end|>\n",
    assistant_prefix="<|im_start|>assistant\n",
    assistant_suffix="<|im_end|>\n",
    system_prefix="<|im_start|>system\n",
    system_suffix="<|im_end|>\n",
    stop_words=["<|im_end|>"],
)

PromptTemplate(
    name="llama3",
    human_prefix="<|start_header_id|>user<|end_header_id|>\n\n",
    human_suffix="<|eot_id|>",
    assistant_prefix="<|start_header_id|>assistant<|end_header_id|>\n\n",
    assistant_suffix="<|eot_id|>",
    system_prefix="<|start_header_id|>system<|end_header_id|>\n\n",
    system_suffix="<|eot_id|>",
    stop_words=["<|eot_id|>"],
)

PromptTemplate(
    name="mistral",
    human_prefix="[INST] ",
    human_suffix=" [/INST]",
    assistant_prefix="",
    assistant_suffix="</s>",
    stop_words=["</s>"],
)

PromptTemplate(
    name="qwen",
    human_prefix="<|im_start|>user\n",
    human_suffix="<|im_end|>\n",
    assistant_prefix="<|im_start|>assistant\n",
    assistant_suffix="<|im_end|>\n",
    stop_words=["<|im_end|>"],
)

PromptTemplate(
    name="phi3",
    human_prefix="<|user|>\n",
    human_suffix="<|end|>\n",
    assistant_prefix="<|assistant|>\n",
    assistant_suffix="<|end|>\n",
    stop_words=["<|end|>"],
)

PromptTemplate(
    name="gemma",
    human_prefix="<start_of_turn>user\n",
    human_suffix="<end_of_turn>\n",
    assistant_prefix="<start_of_turn>model\n",
    assistant_suffix="<end_of_turn>\n",
    stop_words=["<end_of_turn>"],
)


def get_template(name: str) -> PromptTemplate:
    if name not in TEMPLATES:
        raise ValueError(f"Unknown template '{name}'. Available: {list(TEMPLATES)}")
    return TEMPLATES[name]
