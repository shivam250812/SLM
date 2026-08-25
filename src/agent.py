"""Minimal ReAct agent loop with real tool execution.

Extracted verbatim from the experiment notebooks. This exact file is used by
every configuration (FP16, INT4, INT4+QLoRA) -- the model is the only variable.
"""
import re
import time

import torch
from transformers import StoppingCriteria, StoppingCriteriaList


class StopOnObservation(StoppingCriteria):
    def __init__(self, tokenizer, prompt_len, min_new_tokens=20):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.min_new_tokens = min_new_tokens
    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[1] - self.prompt_len < self.min_new_tokens:
            return False
        text = self.tokenizer.decode(
            input_ids[0][self.prompt_len:], skip_special_tokens=True)
        return text.rstrip().endswith("Observation:")


def trim_drift(text):
    for marker in ["---", "**Solution", "**Question", "<|user|>", "<|end|>",
                   "Task:", "\nQuestion:"]:
        idx = text.find(marker)
        if idx > 0:
            text = text[:idx]
    return text


def trim_drift(text):
    for marker in ["---", "**Solution", "**Question", "<|user|>", "<|end|>",
                   "Task:", "\nQuestion:"]:
        idx = text.find(marker)
        if idx > 0:
            text = text[:idx]
    return text


def run_agent(task, tools, system_prompt, model, tokenizer,
              max_steps=6, max_new_tokens=220):
    """Generic ReAct loop with real tool execution.
    Returns (final_answer, steps, tool_calls, latency_s, trace)."""
    prompt = (f"<|system|>\n{system_prompt}<|end|>\n"
              f"<|user|>\nTask: {task}<|end|>\n<|assistant|>\n")
    trace, tool_calls = "", 0
    t0 = time.time()

    for _ in range(max_steps):
        inputs = tokenizer(prompt + trace, return_tensors="pt",
                           truncation=True, max_length=3600).to(model.device)
        plen = inputs.input_ids.shape[1]
        stopping = StoppingCriteriaList([StopOnObservation(tokenizer, plen)])
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                temperature=None, top_p=None, stopping_criteria=stopping,
                pad_token_id=tokenizer.eos_token_id,
            )
        trace += trim_drift(
            tokenizer.decode(out[0][plen:], skip_special_tokens=True))

        if "Final Answer:" in trace or "\nAnswer:" in trace:
            break

        if trace.rstrip().endswith("Observation:"):
            actions = re.findall(r"Action:\s*(\w+)\[(.*?)\]", trace, re.DOTALL)
            if actions:
                name, arg = actions[-1]
                name = name.lower().strip()
                if name in tools:
                    result = tools[name](arg)
                    tool_calls += 1
                    trace = trace.rstrip() + f" {result}\n"
                else:
                    trace = trace.rstrip() + " Error: unknown tool.\n"
            else:
                trace = trace.rstrip() + " Error: no valid Action found.\n"
        else:
            break

    latency = time.time() - t0
    m = re.search(r"Final Answer:\s*(.*?)(?:\n|$)", trace, re.DOTALL)
    if not m:
        m = re.search(r"\nAnswer:\s*(.*?)(?:\n|$)", trace, re.DOTALL)
    final = m.group(1).strip() if m else trace.strip().split("\n")[-1]
    steps = len(re.findall(r"Action:", trace))
    return final, steps, tool_calls, latency, trace
