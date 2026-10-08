"""
STEP 13 (capstone) — Evaluation: prove your agent actually works, systematically.

Concept: so far you've been eyeballing outputs. LangSmith evaluation
formalizes that: define a small DATASET of (input, expected-ish output)
pairs, define an EVALUATOR function that scores each actual output, then
run `evaluate()` — it runs your app against every example and records
scores as an "experiment" you can compare over time in the LangSmith UI.
This is how you'd catch a regression before shipping a prompt change.

Requires the LANGSMITH_API_KEY set in your .env (same free account from
step 05) — this step genuinely needs LangSmith, unlike steps 1-4/6-12
where it was optional.

Run:
  python 13_langsmith_eval.py
Then open smith.langchain.com -> Datasets & Experiments to see the scored results.
"""

import os
from dotenv import load_dotenv
from langsmith import Client
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

if not os.getenv("LANGSMITH_API_KEY") or "your_key_here" in os.getenv("LANGSMITH_API_KEY", ""):
    raise SystemExit(
        "This step needs a real LANGSMITH_API_KEY in your .env (see step 05's docstring for setup). "
        "Every other step in this project works without it — this is the one exception."
    )

# LangSmith reads these same env vars to know where to log the evaluation run.
os.environ["LANGSMITH_TRACING"] = "true"

client = Client()

# --- 1. The thing we're testing: a simple classifier chain ---
llm = ChatGroq(model=MODEL, temperature=0)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Classify the sentiment of the message as exactly one word: positive, negative, or neutral."),
        ("human", "{text}"),
    ]
)
classifier = prompt | llm | StrOutputParser()


def run_classifier(inputs: dict) -> dict:
    return {"label": classifier.invoke({"text": inputs["text"]}).strip().lower()}


# --- 2. A small dataset of known-answer examples ---
DATASET_NAME = "langchain-learning-sentiment"

examples = [
    {"inputs": {"text": "This is the best product I've ever used!"}, "outputs": {"label": "positive"}},
    {"inputs": {"text": "Absolutely terrible experience, would not recommend."}, "outputs": {"label": "negative"}},
    {"inputs": {"text": "It arrived on Tuesday."}, "outputs": {"label": "neutral"}},
    {"inputs": {"text": "I'm so disappointed with how this turned out."}, "outputs": {"label": "negative"}},
]

existing = list(client.list_datasets(dataset_name=DATASET_NAME))
if existing:
    dataset = existing[0]
    print(f"Reusing existing dataset '{DATASET_NAME}'.")
else:
    dataset = client.create_dataset(dataset_name=DATASET_NAME)
    client.create_examples(dataset_id=dataset.id, examples=examples)
    print(f"Created dataset '{DATASET_NAME}' with {len(examples)} examples.")


# --- 3. An evaluator: did the classifier match the expected label? ---
def correct_label(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    return outputs["label"] == reference_outputs["label"]


# --- 4. Run the evaluation ---
if __name__ == "__main__":
    results = client.evaluate(
        run_classifier,
        data=DATASET_NAME,
        evaluators=[correct_label],
        experiment_prefix=f"sentiment-{MODEL.replace('/', '-')}",
        max_concurrency=2,
    )
    print("\nEvaluation complete. Open smith.langchain.com -> Datasets & Experiments -> "
          f"'{DATASET_NAME}' to see per-example scores.")
