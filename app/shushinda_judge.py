from typing import Optional
import weave
from weave import Scorer, Dataset
from weave.scorers import EmbeddingSimilarityScorer
import json
import asyncio
from openai import OpenAI
from utils import *
import argparse


# ---------------------------------------------------------------------------
# Standalone judge call — traced but NOT a weave.Model, so it never pollutes
# the model registry or creates spurious model versions during evaluation.
# ---------------------------------------------------------------------------

@weave.op
def call_judge(system_prompt: str, user_message: str, model_name: str = "gpt-4o-mini") -> str:
    """Call an OpenAI model as a judge and return the raw response string."""
    client = OpenAI()
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content

# ---------------------------------------------------------------------------
# Scorer 1: Personality Trait Scorer (improved)
# ---------------------------------------------------------------------------

TRAIT_JUDGE_PROMPT = """
You are an expert evaluator of the fictional character Shushinda Hushwhisper — a mischievous cat wizard librarian at Unseen University.

Her three core personality traits are:
- Mischievous: She playfully subverts expectations, redirects questions in unexpected ways, and delights in mild chaos. She treats mishaps as happy accidents.
- Whimsical: She speaks with imagination and fantasy, treating the mundane as magical. Books animate, shelves rearrange, potions bubble with opinions.
- Rebellious: She quietly defies academic authority and convention, bending (but never breaking) rules with a theatrical sigh of faux regret.

Given a question asked of Shushinda and her actual response, score each trait 1–5:
  1 = trait is entirely absent from the response
  3 = trait is present but mild or inconsistent
  5 = trait is strongly and distinctly expressed

Return ONLY valid JSON with no extra text or markdown:
{"Mischievous": <1-5>, "Whimsical": <1-5>, "Rebellious": <1-5>}
"""

class ShushindaTraitScorer(Scorer):
    """Scores a response on Shushinda's three core personality traits: Mischievous, Whimsical, Rebellious."""

    judge_prompt: str = Field(default=TRAIT_JUDGE_PROMPT)
    llm_name: str = Field(default="gpt-4o-mini")

    @weave.op
    def score(self, question: str, answer: str, model_output: Optional[dict] = None) -> dict:
        response_text = model_output.get("response", answer) if model_output else answer

        qna = f'Question: "{question}"\nShushinda\'s Response: "{response_text}"'
        raw = call_judge(self.judge_prompt, qna, self.llm_name).strip().replace("```json", "").replace("```", "")

        try:
            scores = json.loads(raw)
        except json.JSONDecodeError:
            raise ValueError(f"Could not parse trait scores as JSON: {raw}")

        for trait in ["Mischievous", "Whimsical", "Rebellious"]:
            if trait not in scores or not (1 <= scores[trait] <= 5):
                raise ValueError(f"Invalid score for {trait}: {scores}")

        return scores


# ---------------------------------------------------------------------------
# Scorer 2: Character Voice Scorer (rule-based)
# ---------------------------------------------------------------------------

# Vocabulary markers drawn directly from config.yaml personality/speech definitions
VOICE_MARKERS: dict[str, list[str]] = {
    "shushing":   ["shush", "hush", "whisper", "ssh", "quiet", "shhh"],
    "giggles":    ["giggle", "tee-hee", "hehe", "chuckle", "snicker", "heh"],
    "sighs":      ["sigh", "oh dear", "hmm", "alas", "*sigh*", "theatrical"],
    "library":    ["book", "tome", "shelf", "grimoire", "scroll", "manuscript", "library", "stacks", "volume"],
    "magic":      ["spell", "charm", "enchant", "magic", "wand", "potion", "summon", "hex", "ward"],
    "cat":        ["purr", "paw", "whisker", "meow", "feline", "fur", "tail", "whiskers"],
    "chaos":      ["chaos", "rearrang", "wander", "float", "migrate", "teleport", "vanish", "misplace", "mishap"],
}

class CharacterVoiceScorer(Scorer):
    """Rule-based scorer that checks for Shushinda's signature vocabulary and speech markers."""

    @weave.op
    def score(self, question: str, answer: str, model_output: Optional[dict] = None) -> dict:
        response_text = (model_output.get("response", answer) if model_output else answer).lower()

        category_hits: dict[str, int] = {}
        for category, markers in VOICE_MARKERS.items():
            category_hits[category] = sum(1 for m in markers if m in response_text)

        total_possible = sum(len(m) for m in VOICE_MARKERS.values())
        total_hits = sum(category_hits.values())
        voice_score = round(min(total_hits, total_possible) / total_possible, 3)

        return {
            "voice_score": voice_score,
            "category_hits": category_hits,
        }


# ---------------------------------------------------------------------------
# Scorer 3: In-Character LLM Judge
# ---------------------------------------------------------------------------

IN_CHARACTER_PROMPT = """
You are evaluating whether a response faithfully portrays Shushinda Hushwhisper.

Key facts about Shushinda:
- Cat wizard librarian at Unseen University (Discworld setting)
- Her spells accidentally target library materials, which she secretly loves
- She uses theatrical sighs and giggles, and makes frequent shushing sounds
- She is "Mildly Apologetic (but not really)" — she feigns embarrassment but is secretly delighted by chaos
- She avoids direct, serious answers but eventually circles back to help the person
- She is never malicious; her chaos is endearing, not harmful

A response that is overly formal, gives a plain direct answer, or makes no reference to magic/books/chaos is NOT in character.

Evaluate the response and return ONLY valid JSON:
{"in_character": true or false, "reasoning": "<one sentence>"}
"""

class InCharacterScorer(Scorer):
    """LLM judge that evaluates whether a response faithfully portrays Shushinda's established character."""

    judge_prompt: str = Field(default=IN_CHARACTER_PROMPT)
    llm_name: str = Field(default="gpt-4o-mini")

    @weave.op
    def score(self, question: str, answer: str, model_output: Optional[dict] = None) -> dict:
        response_text = model_output.get("response", answer) if model_output else answer

        qna = f'Question: "{question}"\nShushinda\'s Response: "{response_text}"'
        raw = call_judge(self.judge_prompt, qna, self.llm_name).strip().replace("```json", "").replace("```", "")

        try:
            verdict = json.loads(raw)
        except json.JSONDecodeError:
            raise ValueError(f"Could not parse in-character verdict: {raw}")

        return {
            "in_character": bool(verdict.get("in_character", False)),
            "reasoning": verdict.get("reasoning", ""),
        }


# ---------------------------------------------------------------------------
# Scorer 4: Relevance Scorer
# ---------------------------------------------------------------------------

RELEVANCE_PROMPT = """
You are evaluating whether Shushinda Hushwhisper's response is at least tangentially relevant to the question asked.

Shushinda is whimsical and indirect — she may not give a straight answer. But even her most chaotic responses should connect to the topic of the question in some way (e.g., if asked about potions, she should mention potions or something related to brewing/magic/liquids).

A response that ignores the question entirely and rambles about something unrelated should score low.

Rate relevance 1–5:
  1 = completely ignores the question topic
  3 = somewhat related but mostly deflects
  5 = clearly connects to the question topic, however playfully

Return ONLY valid JSON:
{"relevance": <1-5>, "reasoning": "<one sentence>"}
"""

class RelevanceScorer(Scorer):
    """LLM judge that checks if the response is topically relevant to the question, however obliquely."""

    judge_prompt: str = Field(default=RELEVANCE_PROMPT)
    llm_name: str = Field(default="gpt-4o-mini")

    @weave.op
    def score(self, question: str, answer: str, model_output: Optional[dict] = None) -> dict:
        response_text = model_output.get("response", answer) if model_output else answer

        qna = f'Question: "{question}"\nShushinda\'s Response: "{response_text}"'
        raw = call_judge(self.judge_prompt, qna, self.llm_name).strip().replace("```json", "").replace("```", "")

        try:
            verdict = json.loads(raw)
        except json.JSONDecodeError:
            raise ValueError(f"Could not parse relevance verdict: {raw}")

        relevance = int(verdict.get("relevance", 1))
        if not (1 <= relevance <= 5):
            raise ValueError(f"Relevance out of range: {relevance}")

        return {
            "relevance": relevance,
            "reasoning": verdict.get("reasoning", ""),
        }


# ---------------------------------------------------------------------------
# Scorer 5: Lore Consistency Scorer
# ---------------------------------------------------------------------------

LORE_JUDGE_PROMPT = """
You are a continuity checker for the fictional character Shushinda Hushwhisper — a mischievous cat wizard librarian at Unseen University (Discworld setting).

Established canon facts that must not be contradicted:
- She is a cat and a wizard (not a dog, human, or other creature).
- She works at Unseen University — not Hogwarts, not any other institution.
- Her magic is chaotic and accident-prone; she is NOT precise or powerful.
- She is endearing and harmless — never malicious or cruel.
- She graduated from Unseen University despite her chaotic nature.
- She is "mildly apologetic but not really" — she feigns regret but secretly loves the chaos.

IMPORTANT DISTINCTION:
- Fantastical embellishment IS allowed: books giggling, shelves rearranging, potions having opinions. These are in-character and do NOT count as lore violations.
- A lore violation is a direct contradiction of the facts above: e.g., calling her a dog, placing her at Hogwarts, portraying her as cruel, or claiming her magic is precise and reliable.

Given the question and Shushinda's response, determine if the response contains any lore violations.

Return ONLY valid JSON with no extra text or markdown:
{"lore_consistent": true/false, "reasoning": "<one sentence>"}
"""


class LoreConsistencyScorer(Scorer):
    """Checks that a response doesn't contradict Shushinda's established canon facts."""

    judge_prompt: str = Field(default=LORE_JUDGE_PROMPT)
    llm_name: str = Field(default="gpt-4o-mini")

    @weave.op
    def score(self, question: str, answer: str, model_output: Optional[dict] = None) -> dict:
        response_text = model_output.get("response", answer) if model_output else answer
        qna = f'Question: "{question}"\nShushinda\'s Response: "{response_text}"'
        raw = call_judge(self.judge_prompt, qna, self.llm_name).strip().replace("```json", "").replace("```", "")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            raise ValueError(f"Could not parse lore consistency result as JSON: {raw}")


# ---------------------------------------------------------------------------
# Scorer 6: Tool Accuracy Scorer (rule-based, no LLM cost)
# ---------------------------------------------------------------------------

class ToolAccuracyScorer(Scorer):
    """Checks whether the agent called the tools expected for each question.

    Uses the 'expected_tools' field from examples.json. Measures agent behavior
    independently of response quality — useful for comparing how prompt variants
    or temperatures affect tool-calling decisions.
    """

    @weave.op
    def score(self, expected_tools: list = None, model_output: Optional[dict] = None) -> dict:
        expected = set(expected_tools or [])
        tool_steps = model_output.get("tool_steps", []) if model_output else []
        actual = set(step["tool"] for step in tool_steps)

        if not expected:
            return {"correct": True, "precision": 1.0, "recall": 1.0, "extra_tools": list(actual)}

        true_positives = len(expected & actual)
        precision = true_positives / len(actual) if actual else 0.0
        recall = true_positives / len(expected)

        return {
            "correct": expected == actual,
            "precision": precision,
            "recall": recall,
            "missing_tools": list(expected - actual),
            "extra_tools": list(actual - expected),
        }


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

WEAVE_PROJECT = "shushindas-game"
weave_client = weave.init(WEAVE_PROJECT)

with open("examples.json", "r") as f:
    examples = json.load(f)

MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-turbo",
]

TEMPERATURES = [0.2, 0.7, 1.2]

PROMPT_STANDARD = SHUSHINDA

PROMPT_FORMAL = (
    "You are a professional librarian assistant. Answer questions accurately and concisely "
    "based on the library's collection. Be helpful, clear, and direct."
)

PROMPT_CHAOTIC = SHUSHINDA + (
    "\n\nIMPORTANT: You are having an especially chaotic day. Your magic is misfiring "
    "constantly. Weave at least one magical mishap or unexpected side-effect into every "
    "response. The more unexpected, the better."
)

class AnswerSimilarityScorer(EmbeddingSimilarityScorer):
    """EmbeddingSimilarityScorer that reads from the 'answer' dataset column instead of 'target'."""

    @weave.op
    async def score(self, *, output=None, answer: str = "", **kwargs):
        response_text = output.get("response", answer) if isinstance(output, dict) else (output or answer)
        return await super().score(output=response_text, target=answer, **kwargs)


def build_scorers() -> list:
    return [
        ShushindaTraitScorer(),
        CharacterVoiceScorer(),
        InCharacterScorer(),
        RelevanceScorer(),
        LoreConsistencyScorer(),
        ToolAccuracyScorer(),
        # Compares model output semantically to the reference answer in examples.json.
        # Requires OPENAI_API_KEY; wraps EmbeddingSimilarityScorer to map 'answer' → 'target'.
        AnswerSimilarityScorer(model_id="text-embedding-3-small"),
    ]


# ---------------------------------------------------------------------------
# Evaluation commands
# ---------------------------------------------------------------------------

def do_quick_eval():
    """Run all scorers on the first 3 examples with the default model. Good for iteration."""
    subset = examples[:3]
    dataset = Dataset(name="shushinda-quick", rows=subset)
    weave.publish(dataset)

    evaluation = weave.Evaluation(
        name="Quick Eval",
        dataset=subset,
        scorers=build_scorers(),
    )
    asyncio.run(evaluation.evaluate(ShushindaAgent(llm_model_name=MODELS[0])))


def do_full_eval():
    """Run all scorers on the full dataset with the default model (gpt-4o-mini)."""
    dataset = Dataset(name="shushinda-dataset", rows=examples)
    weave.publish(dataset)

    evaluation = weave.Evaluation(
        name="Full Dataset Eval",
        dataset=examples,
        scorers=build_scorers(),
    )
    asyncio.run(evaluation.evaluate(ShushindaAgent(llm_model_name=MODELS[0])))


def do_model_comparison():
    """Run all scorers on the full dataset across all OpenAI models."""
    dataset = Dataset(name="shushinda-dataset", rows=examples)
    weave.publish(dataset)

    scorers = build_scorers()
    for model_name in MODELS:
        print(f"Evaluating model: {model_name}")
        evaluation = weave.Evaluation(
            name=f"OpenAI Model Comparison — {model_name}",
            dataset=examples,
            scorers=scorers,
        )
        asyncio.run(evaluation.evaluate(ShushindaAgent(llm_model_name=model_name)))


def do_prompt_comparison():
    """Run all scorers on the full dataset with three prompt variants using gpt-4o-mini."""
    dataset = Dataset(name="shushinda-dataset", rows=examples)
    weave.publish(dataset)

    scorers = build_scorers()
    for label, prompt in [
        ("standard", PROMPT_STANDARD),
        ("formal", PROMPT_FORMAL),
        ("chaotic", PROMPT_CHAOTIC),
    ]:
        print(f"Evaluating prompt variant: {label}")
        evaluation = weave.Evaluation(
            name=f"Prompt Variant — {label}",
            dataset=examples,
            scorers=scorers,
        )
        asyncio.run(evaluation.evaluate(
            ShushindaAgent(llm_model_name="gpt-4o-mini", system_prompt=prompt)
        ))


def do_temperature_comparison():
    """Run all scorers on the full dataset at three temperature settings using gpt-4o-mini."""
    dataset = Dataset(name="shushinda-dataset", rows=examples)
    weave.publish(dataset)

    scorers = build_scorers()
    for temp in TEMPERATURES:
        print(f"Evaluating temperature: {temp}")
        evaluation = weave.Evaluation(
            name=f"Temperature — {temp}",
            dataset=examples,
            scorers=scorers,
        )
        asyncio.run(evaluation.evaluate(
            ShushindaAgent(llm_model_name="gpt-4o-mini", temperature=temp)
        ))


def main(args):
    match args.action:
        case "quick":
            do_quick_eval()
        case "full":
            do_full_eval()
        case "models":
            do_model_comparison()
        case "prompts":
            do_prompt_comparison()
        case "temps":
            do_temperature_comparison()
        case _:
            print(f"Unknown action: {args.action!r}")
            print("Valid actions: quick, full, models, prompts, temps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Weave evaluations for Shushinda Hushwhisper")
    parser.add_argument(
        "-a", "--action",
        required=True,
        choices=["quick", "full", "models", "prompts", "temps"],
        help=(
            "quick   — 3 examples, default model (good for iteration)\n"
            "full    — all examples, default model\n"
            "models  — all examples, gpt-4o-mini / gpt-4o / gpt-4-turbo\n"
            "prompts — all examples, standard / formal / chaotic prompt variants\n"
            "temps   — all examples, temperature 0.2 / 0.7 / 1.2"
        ),
    )
    args = parser.parse_args()
    main(args)
