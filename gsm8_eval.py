import os
import re
import time

import tiktoken
from datasets import load_dataset
from loguru import logger
from swarms import Agent, OpenAIChat
from swarms_evals.gsm8k_system import GSM8K_PROMPT
from dotenv import load_dotenv


load_dotenv()

# Initialize logger
logger.add(
    "metrics.log", format="{time} {level} {message}", level="INFO"
)


# Normalization function
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punctuation(text):
        return re.sub(r"[^a-zA-Z0-9\s]", "", text)

    def lower(text):
        return text.lower()

    return white_space_fix(
        remove_articles(remove_punctuation(lower(s)))
    )


# Load GSM8K dataset
def load_gsm8k_dataset():
    dataset = load_dataset("gsm8k", "main")
    return dataset["test"]


# Tokenizer for counting tokens
def count_tokens(text: str = None):
    tokenizer = tiktoken.encoding_for_model("gpt-4o")
    return len(tokenizer.encode(text))


# Evaluate model on GSM8K
def evaluate_model_on_gsm8k(agent: Agent, test_data):
    correct = 0
    total = len(test_data)
    total_tokens = 0
    total_time = 0

    for example in test_data:
        question = example["question"]
        answer = normalize_answer(example["answer"])

        start_time = time.time()

        # Run the agent on the question
        predicted_answer = agent.run((question,))

        end_time = time.time()
        latency = end_time - start_time
        total_time += latency

        predicted_answer = normalize_answer(predicted_answer)

        # Count tokens
        tokens = count_tokens(question + predicted_answer)
        total_tokens += tokens

        if predicted_answer == answer:
            correct += 1

        # Log metrics
        logger.info(f"Question: {question}")
        logger.info(f"Predicted Answer: {predicted_answer}")
        logger.info(f"Correct Answer: {answer}")
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


# Initialize the agent with ChromaDB memory

agent = Agent(
    agent_name="GsM8K-Agent",
    system_prompt=GSM8K_PROMPT,
    llm=OpenAIChat(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    ),
    max_loops=1,
    autosave=True,
    verbose=True,
)

# Load the GSM8K dataset
test_data = load_gsm8k_dataset()

# Evaluate the agent on GSM8K
accuracy, avg_latency, avg_tokens = evaluate_model_on_gsm8k(
    agent, test_data
)

print(f"GSM8K Evaluation Accuracy: {accuracy * 100:.2f}%")
print(f"Average Latency: {avg_latency:.2f} seconds")
print(f"Average Tokens: {avg_tokens}")
