import os
import re
import time

import tiktoken
from datasets import load_dataset
from loguru import logger
from swarms import Agent
from dotenv import load_dotenv

load_dotenv()

# Initialize logger
logger.add("metrics.log", format="{time} {level} {message}", level="INFO")

# Normalization function
def normalize_answer(s):
    """Lower text and remove punctuation, articles, and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punctuation(text):
        return re.sub(r"[^a-zA-Z0-9\s]", "", text)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punctuation(lower(s))))

# Load HellaSwag dataset
def load_hellaswag_dataset():
    dataset = load_dataset("hellaswag")
    return dataset["validation"]  # Use the validation set for evaluation

# Tokenizer for counting tokens
def count_tokens(text: str = None):
    tokenizer = tiktoken.encoding_for_model("gpt-4")
    return len(tokenizer.encode(text))

# Evaluate model on HellaSwag
def evaluate_model_on_hellaswag(agent: Agent, test_data):
    correct = 0
    total = len(test_data)
    total_tokens = 0
    total_time = 0

    for example in test_data:
        context = example["context"]
        ending_options = example["endings"]
        correct_ending = ending_options[example["label"]]

        start_time = time.time()

        # Run the agent on the context with each ending option
        predicted_ending = agent.run((context, ending_options))

        end_time = time.time()
        latency = end_time - start_time
        total_time += latency

        predicted_ending = normalize_answer(predicted_ending)
        correct_ending = normalize_answer(correct_ending)

        # Count tokens
        tokens = count_tokens(context + predicted_ending)
        total_tokens += tokens

        if predicted_ending == correct_ending:
            correct += 1

        # Log metrics
        logger.info(f"Context: {context}")
        logger.info(f"Predicted Ending: {predicted_ending}")
        logger.info(f"Correct Ending: {correct_ending}")
        logger.info(f"Latency: {latency:.2f} seconds")
        logger.info(f"Tokens: {tokens}")

    accuracy = correct / total
    avg_latency = total_time / total
    avg_tokens = total_tokens / total

    # Log final metrics
    logger.info(f"Final Accuracy: {accuracy * 100:.2f}%")
    logger.info(f"Average Latency: {avg_latency:.2f} seconds")
    logger.info(f"Average Tokens: {avg_tokens}")

    return accuracy, avg_latency, avg_tokens

system_prompt = """
You are an AI assistant tasked with completing given contexts by selecting the most appropriate ending from a list of provided options. Each context describes a scenario, and you will choose the ending that best fits the context logically and naturally.

When given a context and a list of possible endings, evaluate each ending carefully, considering coherence, relevance, and overall fit with the context. Select the ending that provides the most logical and contextually appropriate conclusion to the scenario.
"""

# Initialize the agent with ChromaDB memory
agent = Agent(
    agent_name="HellaSwag-Agent",
    system_prompt=system_prompt,
    llm=OpenAIChat(openai_api_key=os.getenv("OPENAI_API_KEY")),
    max_loops=1,
    autosave=True,
    verbose=True,
)

# Load the HellaSwag dataset
test_data = load_hellaswag_dataset()

print(len(test_data))

# Evaluate the agent on HellaSwag
accuracy, avg_latency, avg_tokens = evaluate_model_on_hellaswag(agent, test_data)

print(f"HellaSwag Evaluation Accuracy: {accuracy * 100:.2f}%")
print(f"Average Latency: {avg_latency:.2f} seconds")
print(f"Average Tokens: {avg_tokens}")
