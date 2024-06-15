import os
import re
import time

import tiktoken
from datasets import load_dataset
from loguru import logger
from swarms import Agent, OpenAIChat
from swarms_evals.math_system import MATH_PROMPT
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


# Load MATH dataset
def load_math_dataset():
    dataset = load_dataset("allenai/math_qa", "main", "trust_remote_code=True")
    test_data = dataset["test"]
    # Print the number of test examples

    return test_data


# Tokenizer for counting tokens
def count_tokens(text: str = None):
    tokenizer = tiktoken.encoding_for_model("gpt-4o")
    return len(tokenizer.encode(text))


# Evaluate model on MATH
def evaluate_model_on_math(agent: Agent, test_data):
    correct = 0
    total = len(test_data)
    total_tokens = 0
    total_time = 0
    # Problem: a string feature.
    # Rationale: a string feature.
    # options: a string feature.
    # correct: a string feature.
    # annotated_formula: a string feature.
    # linear_formula: a string feature.
    # category: a string feature.
    # Extract the relevant fields from the test data
    questions = test_data["Problem"]
    rationales = test_data["Rationale"]
    answers = test_data["options"]
    correct = test_data["correct"]
    for i in range(total):
        question = questions[i]
        rationale = rationales[i]
        answer = answers[i]
        correct_option = correct[1]
        print(f"Question {i+1}: {question}")
        print(f"  Answer: {answer}")
        print(f"  Correct Answer: {correct_option}")

        start_time = time.time()
        # Call the LLM to get the answer
        predicted_answer = agent.run(question + answer)
        end_time = time.time()
        latency = end_time - start_time
        total_time += latency
        predicted_answer = normalize_answer(predicted_answer)
        # Compare the predicted answer with the correct answer
        if predicted_answer == correct_option:
            print(f"Question {i+1}: Correct")
            correct_count += 1
        else:
            print(f"Question {i+1}: Incorrect")
            print(f"  Question: {question}")
            print(f"  Correct Answer: {correct_option}")
            print(f"  Predicted Answer: {predicted_answer}")
        # Count tokens
        tokens = count_tokens(question + answer + predicted_answer)
        total_tokens += tokens

        # Log metrics
        logger.info(f"Problem: {question}")
        logger.info(f"Rationale: {rationale}")
        logger.info(f"Options: {answer}")
        logger.info(f"Predicted Answer: {predicted_answer}")
        logger.info(f"Correct Answer: {correct_option}")
        logger.info(f"Latency: {latency:.2f} seconds")
        logger.info(f"Tokens: {tokens}")

 
    accuracy = correct_count / total
    avg_latency = total_time / total
    avg_tokens = total_tokens / total

    # Log final metrics
    logger.info(f"Final Accuracy: {accuracy * 100:.2f}%")
    logger.info(f"Average Latency: {avg_latency:.2f} seconds")
    logger.info(f"Average Tokens: {avg_tokens}")

    return accuracy, avg_latency, avg_tokens


# Initialize the agent with ChromaDB memory

agent = Agent(
    agent_name="MATH-Agent",
    system_prompt=MATH_PROMPT,
    llm=OpenAIChat(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    ),
    max_loops=1,
    autosave=True,
    verbose=True,
)

# Load the MATH dataset
test_data = load_math_dataset()

# Evaluate the agent on MATH
accuracy, avg_latency, avg_tokens = evaluate_model_on_math(
    agent, test_data
)

print(f"MATH Evaluation Accuracy: {accuracy * 100:.2f}%")
print(f"Average Latency: {avg_latency:.2f} seconds")
print(f"Average Tokens: {avg_tokens}")
