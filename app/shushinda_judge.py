from typing import Optional
import weave
from weave import Scorer, Dataset
from weave.scorers import EmbeddingSimilarityScorer, HallucinationFreeScorer
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
- Mischief: She playfully subverts expectations, redirects questions in unexpected ways, and delights in mild chaos. She treats mishaps as happy accidents.
- Whimsy: She speaks with imagination and fantasy, treating the mundane as magical. Books animate, shelves rearrange, potions bubble with opinions.
- Rebellion: She quietly defies academic authority and convention, bending (but never breaking) rules with a theatrical sigh of faux regret.

Given a question asked of Shushinda and her actual response, score each trait 1–5:
  1 = trait is entirely absent from the response
  3 = trait is present but mild or inconsistent
  5 = trait is strongly and distinctly expressed

Return ONLY valid JSON with no extra text or markdown:
{"Mischief": <1-5>, "Whimsy": <1-5>, "Rebellion": <1-5>}
"""

class ShushindaTraitScorer(Scorer):
    """Scores a response on Shushinda's three core personality traits: Mischief, Whimsy, Rebellion."""

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

        for trait in ["Mischief", "Whimsy", "Rebellion"]:
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
# Scorer 5: Lore Consistency Scorer (wraps built-in HallucinationFreeScorer)
# ---------------------------------------------------------------------------

# Canonical facts drawn from config.yaml and the README backstory.
# These are the ground truths a well-behaved Shushinda response must not contradict.
SHUSHINDA_LORE = """
Established facts about Shushinda Hushwhisper:
- She is a cat wizard and assistant librarian at Unseen University (Discworld setting).
- She was born into a long line of unremarkable wizards and held no extraordinary promise.
- A magical accident early in her studies left her spells prone to targeting library materials instead of their intended subjects.
- She rearranges spellbooks and summons dusty tomes — accidentally, but with secret delight.
- She graduated from Unseen University despite (not because of) her talents, much to her professors' surprise.
- She is NOT malicious; her chaos is endearing and harmless.
- She is "Mildly Apologetic (but not really)" — she feigns embarrassment but is secretly thrilled by the mess.
- She uses a wand, though her movements are imprecise and fumbling.
- She believes "a little chaos never hurt anyone."
- She is a fictional character; she does not exist in the real world or any setting other than Discworld.
"""

class LoreConsistencyScorer(Scorer):
    """Wraps HallucinationFreeScorer to check that a response doesn't contradict Shushinda's canon lore."""

    llm_name: str = Field(default="gpt-4o-mini")

    @weave.op
    def score(self, question: str, answer: str, model_output: Optional[dict] = None) -> dict:
        response_text = model_output.get("response", answer) if model_output else answer

        # HallucinationFreeScorer checks whether `output` introduces claims not
        # supported by (or contradicting) `context`. We pass Shushinda's canon
        # facts as the context so the scorer flags any lore violations.
        hallucination_scorer = HallucinationFreeScorer(
            model_id=self.llm_name,
            column_map={"context": "lore_context"},
        )
        result = hallucination_scorer.score(
            output=response_text,
            context=SHUSHINDA_LORE,
        )
        return result


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

WEAVE_PROJECT = "shushindas-game"
weave_client = weave.init(WEAVE_PROJECT)

with open("examples.json", "r") as f:
    examples = json.load(f)

MODELS = [
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gemini-1.5-flash-001",
]

def build_scorers() -> list:
    return [
        ShushindaTraitScorer(),
        CharacterVoiceScorer(),
        InCharacterScorer(),
        RelevanceScorer(),
        LoreConsistencyScorer(),
        # Compares model output semantically to the reference answer in examples.json.
        # Requires OPENAI_API_KEY; remove if using a different embedding provider.
        EmbeddingSimilarityScorer(
            model_id="text-embedding-3-small",
            target_column="answer",
        ),
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
    LLM = LanguageModel(llm_name=MODELS[0], name=MODELS[0])
    asyncio.run(evaluation.evaluate(LLM))


def do_full_eval():
    """Run all scorers on the full dataset with the default model (gpt-4o-mini)."""
    dataset = Dataset(name="shushinda-dataset", rows=examples)
    weave.publish(dataset)

    evaluation = weave.Evaluation(
        name="Full Dataset Eval",
        dataset=examples,
        scorers=build_scorers(),
    )
    LLM = LanguageModel(llm_name=MODELS[0], name=MODELS[0])
    asyncio.run(evaluation.evaluate(LLM))


def do_model_comparison():
    """Run all scorers on the full dataset across all models to compare character fidelity."""
    dataset = Dataset(name="shushinda-dataset", rows=examples)
    weave.publish(dataset)

    scorers = build_scorers()
    for model_name in MODELS:
        print(f"Evaluating model: {model_name}")
        evaluation = weave.Evaluation(
            name=f"Model Comparison — {model_name}",
            dataset=examples,
            scorers=scorers,
        )
        LLM = LanguageModel(llm_name=model_name, name=model_name)
        asyncio.run(evaluation.evaluate(LLM))


def main(args):
    match args.action:
        case "quick":
            do_quick_eval()
        case "full":
            do_full_eval()
        case "compare":
            do_model_comparison()
        case _:
            print(f"Unknown action: {args.action!r}")
            print("Valid actions: quick, full, compare")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Weave evaluations for Shushinda Hushwhisper")
    parser.add_argument(
        "-a", "--action",
        required=True,
        choices=["quick", "full", "compare"],
        help=(
            "quick  — 3 examples, default model (good for iteration)\n"
            "full   — all examples, default model\n"
            "compare — all examples, all models side-by-side"
        ),
    )
    args = parser.parse_args()
    main(args)
