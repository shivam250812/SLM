"""Tools available to the agent: a calculator and a benchmark-specific search."""
import re

def tool_calculator(expr):
    try:
        expr = expr.replace("^", "**").replace(",", "").replace("$", "")
        expr = expr.replace("%", "/100")
        if not re.fullmatch(r"[0-9+\-*/().\s]+", expr):
            return "Error: invalid expression."
        result = eval(expr, {"__builtins__": {}}, {})
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return str(round(result, 6) if isinstance(result, float) else result)
    except Exception as e:
        return f"Error: {e}"


KNOWLEDGE_BASE = {
    # --- original facts ---
    "population of paris": "The population of Paris is 2.1 million.",
    "population of tokyo": "The population of Tokyo is 14 million.",
    "population of new delhi": "The population of New Delhi is 32 million.",
    "population of delhi": "The population of New Delhi is 32 million.",
    "capital of japan": "The capital of Japan is Tokyo.",
    "capital of india": "The capital of India is New Delhi.",
    "capital of france": "The capital of France is Paris.",
    "ceo of microsoft": "The CEO of Microsoft is Satya Nadella.",
    "phi-3": "Phi-3-mini has 3.8 billion parameters.",
    "parameters": "Phi-3-mini has 3.8 billion parameters.",
    "speed of light": "The speed of light is 299792 km/s.",
    "tallest mountain": "The tallest mountain is Mount Everest at 8849 meters.",
    "mount everest": "The tallest mountain is Mount Everest at 8849 meters.",
    "longest river": "The longest river is the Nile at 6650 km.",
    "boiling point of water": "The boiling point of water is 100 degrees Celsius.",
    # --- new facts for the 25 new tasks ---
    "capital of canada": "The capital of Canada is Ottawa.",
    "population of ottawa": "The population of Ottawa is 1.4 million.",
    "ceo of apple": "The CEO of Apple is Tim Cook.",
    "largest desert": "The largest desert is the Sahara at 9.2 million square km.",
    "population of sydney": "The population of Sydney is 5.3 million.",
    "population of cairo": "The population of Cairo is 22 million.",
    "continents": "There are 7 continents on Earth.",
    "distance from earth to the moon": "The distance from Earth to the Moon is 384400 km.",
    "earth to the moon": "The distance from Earth to the Moon is 384400 km.",
    "body temperature": "The normal human body temperature is 37 degrees Celsius.",
    "capital of brazil": "The capital of Brazil is Brasilia.",
    "largest ocean": "The largest ocean is the Pacific Ocean.",
}

def tool_search(query):
    q = query.lower().strip()
    for key, val in KNOWLEDGE_BASE.items():
        if key in q:
            return val
    best, best_score = None, 0
    for key, val in KNOWLEDGE_BASE.items():
        score = len(set(key.split()) & set(q.split()))
        if score > best_score:
            best, best_score = val, score
    return best if best else "No results found."


def make_hotpot_search(item):
    """Per-question search tool over that question's 10 paragraphs."""
    def search(query):
        q_words = set(re.findall(r"\w+", query.lower()))
        scored = []
        for title, para in zip(item["titles"], item["paragraphs"]):
            t_words = set(re.findall(r"\w+", (title + " " + para).lower()))
            title_words = set(re.findall(r"\w+", title.lower()))
            score = len(q_words & t_words) + 3 * len(q_words & title_words)
            scored.append((score, title, para))
        scored.sort(key=lambda x: -x[0])
        best = scored[0]
        if best[0] == 0:
            return "No results found."
        return f"[{best[1]}] {best[2][:600]}"
    return search
