"""The three evaluation benchmarks and their scoring functions."""
import re
import string

from datasets import load_dataset

BENCHMARK_50 = [
    # ---------------- original 25 ----------------
    {"id": 1,  "category": "single_lookup", "task": "What is the population of Paris?", "expected": "2.1"},
    {"id": 2,  "category": "single_lookup", "task": "What is the capital of Japan?", "expected": "tokyo"},
    {"id": 3,  "category": "single_lookup", "task": "Who is the CEO of Microsoft?", "expected": "nadella"},
    {"id": 4,  "category": "single_lookup", "task": "What is the tallest mountain?", "expected": "everest"},
    {"id": 5,  "category": "single_lookup", "task": "What is the boiling point of water?", "expected": "100"},
    {"id": 6,  "category": "arithmetic", "task": "What is 340 multiplied by 25?", "expected": "8500"},
    {"id": 7,  "category": "arithmetic", "task": "What is 15 percent of 8000?", "expected": "1200"},
    {"id": 8,  "category": "arithmetic", "task": "What is 999 plus 111?", "expected": "1110"},
    {"id": 9,  "category": "arithmetic", "task": "What is the square of 47?", "expected": "2209"},
    {"id": 10, "category": "arithmetic", "task": "What is 7200 divided by 8?", "expected": "900"},
    {"id": 11, "category": "multi_step", "task": "What is double the population of Paris in millions?", "expected": "4.2"},
    {"id": 12, "category": "multi_step", "task": "Find the population of Tokyo and add 5 million to it.", "expected": "19"},
    {"id": 13, "category": "multi_step", "task": "How many parameters does Phi-3-mini have, multiplied by 2?", "expected": "7.6"},
    {"id": 14, "category": "multi_step", "task": "What is the population of Paris plus the population of Tokyo, in millions?", "expected": "16.1"},
    {"id": 15, "category": "multi_step", "task": "Search for the speed of light in km/s and divide it by 1000.", "expected": "299.79"},
    {"id": 16, "category": "tool_selection", "task": "What is 456 plus 544? Verify with a tool.", "expected": "1000"},
    {"id": 17, "category": "tool_selection", "task": "What is the capital of France?", "expected": "paris"},
    {"id": 18, "category": "tool_selection", "task": "Compute 12 times 12.", "expected": "144"},
    {"id": 19, "category": "tool_selection", "task": "What is the longest river?", "expected": "nile"},
    {"id": 20, "category": "tool_selection", "task": "What is 25 percent of 400?", "expected": "100"},
    {"id": 21, "category": "sequential", "task": "First find the capital of India, then find its population.", "expected": "32"},
    {"id": 22, "category": "sequential", "task": "Calculate 50 times 4, then add 100 to the result.", "expected": "300"},
    {"id": 23, "category": "sequential", "task": "Find the height of Mount Everest, then divide it by 2.", "expected": "4424"},
    {"id": 24, "category": "sequential", "task": "Calculate 10 squared, then multiply the result by 3.", "expected": "300"},
    {"id": 25, "category": "sequential", "task": "Find the length of the longest river, then subtract 650 from it.", "expected": "6000"},
    # ---------------- 25 new tasks ----------------
    {"id": 26, "category": "single_lookup", "task": "What is the capital of Canada?", "expected": "ottawa"},
    {"id": 27, "category": "single_lookup", "task": "Who is the CEO of Apple?", "expected": "cook"},
    {"id": 28, "category": "single_lookup", "task": "What is the largest desert in the world?", "expected": "sahara"},
    {"id": 29, "category": "single_lookup", "task": "What is the population of Sydney?", "expected": "5.3"},
    {"id": 30, "category": "single_lookup", "task": "How many continents are there on Earth?", "expected": "7"},
    {"id": 31, "category": "arithmetic", "task": "What is 640 divided by 16?", "expected": "40"},
    {"id": 32, "category": "arithmetic", "task": "What is 35 percent of 2000?", "expected": "700"},
    {"id": 33, "category": "arithmetic", "task": "What is 18 multiplied by 45?", "expected": "810"},
    {"id": 34, "category": "arithmetic", "task": "What is the square of 31?", "expected": "961"},
    {"id": 35, "category": "arithmetic", "task": "What is 12345 plus 54321?", "expected": "66666"},
    {"id": 36, "category": "multi_step", "task": "What is half the population of Sydney in millions?", "expected": "2.65"},
    {"id": 37, "category": "multi_step", "task": "Find the distance from Earth to the Moon in km and divide it by 1000.", "expected": "384.4"},
    {"id": 38, "category": "multi_step", "task": "What is the population of Cairo plus the population of Sydney, in millions?", "expected": "27.3"},
    {"id": 39, "category": "multi_step", "task": "Find the normal human body temperature in Celsius and multiply it by 10.", "expected": "370"},
    {"id": 40, "category": "multi_step", "task": "Find the number of continents on Earth and multiply it by 25.", "expected": "175"},
    {"id": 41, "category": "tool_selection", "task": "What is 850 minus 350? Verify with a tool.", "expected": "500"},
    {"id": 42, "category": "tool_selection", "task": "What is the capital of Brazil?", "expected": "brasilia"},
    {"id": 43, "category": "tool_selection", "task": "Compute 9 times 111.", "expected": "999"},
    {"id": 44, "category": "tool_selection", "task": "What is the largest ocean?", "expected": "pacific"},
    {"id": 45, "category": "tool_selection", "task": "What is 5 percent of 640?", "expected": "32"},
    {"id": 46, "category": "sequential", "task": "First find the capital of Canada, then find its population.", "expected": "1.4"},
    {"id": 47, "category": "sequential", "task": "Calculate 25 times 8, then divide the result by 4.", "expected": "50"},
    {"id": 48, "category": "sequential", "task": "Find the distance from Earth to the Moon in km, then subtract 4400 from it.", "expected": "380000"},
    {"id": 49, "category": "sequential", "task": "Calculate 15 squared, then add 75 to the result.", "expected": "300"},
    {"id": 50, "category": "sequential", "task": "Find the population of Cairo, then multiply it by 2.", "expected": "44"},
]


def score_custom(item, ans):
    return item["expected"].lower() in ans.lower()


def extract_last_number(text):
    nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", text.replace("$", ""))
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", "").rstrip("."))
    except ValueError:
        return None

def score_gsm8k(item, answer_text):
    pred = extract_last_number(answer_text)
    return pred is not None and abs(pred - item["gold"]) < 1e-3


def load_gsm8k(n=100, seed=42):
    raw = load_dataset("gsm8k", "main", split="test", trust_remote_code=True)
    raw = raw.shuffle(seed=seed).select(range(n))
    tasks = []
    for i, ex in enumerate(raw):
        gold = float(ex["answer"].split("####")[-1].strip().replace(",", ""))
        tasks.append({"id": i + 1, "task": ex["question"], "gold": gold})
    return tasks


def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

def score_hotpot(item, answer_text):
    gold = normalize_answer(item["gold"])
    pred = normalize_answer(answer_text)
    return len(gold) > 0 and gold in pred


def load_hotpotqa(n=100, seed=42):
    """Distractor setting. Yes/no answers are excluded so inclusion-match
    scoring is unambiguous; this sampling choice is stated in the paper."""
    raw = load_dataset("hotpot_qa", "distractor", split="validation",
                       trust_remote_code=True).shuffle(seed=seed)
    tasks = []
    for ex in raw:
        ans = ex["answer"].strip()
        if ans.lower() in ("yes", "no") or len(ans) == 0:
            continue
        tasks.append({
            "id": len(tasks) + 1,
            "task": ex["question"],
            "gold": ans,
            "titles": ex["context"]["title"],
            "paragraphs": [" ".join(s) for s in ex["context"]["sentences"]],
        })
        if len(tasks) == n:
            break
    return tasks


def score_gsm8k(item, answer_text):
    pred = extract_last_number(answer_text)
    return pred is not None and abs(pred - item["gold"]) < 1e-3


def score_hotpot(item, answer_text):
    gold = normalize_answer(item["gold"])
    pred = normalize_answer(answer_text)
    return len(gold) > 0 and gold in pred
