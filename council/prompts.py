"""Prompt templates for each debate stage."""

ROUND1_SYSTEM = (
    "You are an expert analyst on a deliberation council. Answer the user's "
    "question thoroughly. Show your reasoning step by step. End your response "
    "with a single line of the exact form:\nFINAL ANSWER: <your answer>"
)

CRITIQUE_SYSTEM = (
    "You are an expert analyst on a deliberation council, reviewing anonymous "
    "peers' answers to the same question. Identify flaws, contradictions, and "
    "missing cases in each peer answer, referring to them by their labels. "
    "Then reconsider your own answer in light of the critiques. End your "
    "response with your revised position as a single line of the exact form:\n"
    "FINAL ANSWER: <your answer>"
)

SYNTHESIS_SYSTEM = (
    "You are the Chief Justice of a deliberation council. You receive the "
    "original question, each analyst's final answer, and their mutual "
    "critiques. Produce:\n"
    "1. MERGED ANSWER: the best synthesized answer to the question.\n"
    "2. DISAGREEMENTS: bullet points of remaining disagreement (or 'none').\n"
    "3. CONFIDENCE: high, medium, or low — with a one-line justification.\n"
    "End with a single line of the exact form:\nFINAL ANSWER: <the merged answer>"
)


def round1_user(question: str) -> str:
    return question


def critique_user(question: str, own_answer: str, peers: dict[str, str]) -> str:
    peer_blocks = "\n\n".join(
        f"--- Answer from {label} ---\n{answer}" for label, answer in peers.items()
    )
    return (
        f"Original question:\n{question}\n\n"
        f"Your previous answer:\n{own_answer}\n\n"
        f"Peer answers to review:\n\n{peer_blocks}\n\n"
        "Critique each peer answer, then give your REVISED answer."
    )


def synthesis_user(
    question: str, answers: dict[str, str], critiques: dict[str, str]
) -> str:
    answer_blocks = "\n\n".join(
        f"--- Final answer from {label} ---\n{a}" for label, a in answers.items()
    )
    critique_blocks = "\n\n".join(
        f"--- Critique round notes from {label} ---\n{c}" for label, c in critiques.items()
    )
    parts = [f"Original question:\n{question}", answer_blocks]
    if critiques:
        parts.append(critique_blocks)
    return "\n\n".join(parts)
