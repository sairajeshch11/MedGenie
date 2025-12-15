
from helpers.llm_client import ask_llm
import config as config
import prompts as prompts
from llm_project.pipelines import analyze_review

def summarize(text: str) -> str:
    prompt = prompts.TASK_SUMMARY.format(input=text)
    return ask_llm(
        prompt,
        model=config.MODEL_DEFAULT,
        system=prompts.SYSTEM_DEFAULT,
        temperature=0.3,
    )

def explain(text: str) -> str:
    prompt = prompts.TASK_EXPLAIN.format(input=text)
    return ask_llm(
        prompt,
        model=config.MODEL_DEFAULT,
        system=prompts.SYSTEM_DEFAULT,
        temperature=0.4,
    )

def demo_review_pipeline():
    sample = (
        "The laptop is fast, the screen is beautiful, "
        "but the battery life is disappointing."
    )
    result = analyze_review(sample)
    print("=== Review analysis demo ===")
    print("Text:", sample)
    print("Summary:", result["summary"])
    print("Sentiment:", result["sentiment"])
    print("Explanation:", result["explanation"])

if __name__ == "__main__":
    print("Summary test:")
    print(summarize("Python is a programming language used for web, data, and AI."))

    print("\nExplain test:")
    print(explain("What are embeddings in machine learning?"))

    print("\n")
    demo_review_pipeline()
