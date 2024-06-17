# MEASURING MASSIVE MULTITASK LANGUAGE UNDERSTANDING
# https://arxiv.org/pdf/2009.03300 pdf

# example
# {
#   "question": "What is the embryological origin of the hyoid bone?",
#   "choices": ["The first pharyngeal arch", "The first and second pharyngeal arches", "The second pharyngeal arch", "The second and third pharyngeal arches"],
#   "answer": "D"
# }

import os
import re
import time

import tiktoken
from datasets import load_dataset
from loguru import logger
from swarms import Agent, OpenAIChat
from swarms_evals.mmlu_system import MMLU_PROMPT
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


# Load MMLU dataset
def load_mmlu_dataset():
    # You need to specify the tasks to download:
    #  ['abstract_algebra', 'anatomy', 'astronomy', 
    # 'business_ethics', 'clinical_knowledge', 
    # 'college_biology', 'college_chemistry', 
    # 'college_computer_science', 'college_mathematics', 
    # 'college_medicine', 'college_physics', 'computer_security', 
    # 'conceptual_physics', 'econometrics', 'electrical_engineering', 
    # 'elementary_mathematics', 'formal_logic', 'global_facts', 
    # 'high_school_biology', 'high_school_chemistry', 
    # 'high_school_computer_science', 'high_school_european_history', 
    # 'high_school_geography', 'high_school_government_and_politics', 
    # 'high_school_macroeconomics', 'high_school_mathematics', 
    # 'high_school_microeconomics', 'high_school_physics', 
    # 'high_school_psychology', 'high_school_statistics', 
    # 'high_school_us_history', 'high_school_world_history', 
    # 'human_aging', 'human_sexuality', 'international_law', 
    # 'jurisprudence', 'logical_fallacies', 'machine_learning', 
    # 'management', 'marketing', 'medical_genetics', 
    # 'miscellaneous', 'moral_disputes', 'moral_scenarios', 
    # 'nutrition', 'philosophy', 'prehistory', 
    # 'professional_accounting', 'professional_law', 
    # 'professional_medicine', 'professional_psychology', 
    # 'public_relations', 'security_studies', 'sociology', 
    # 'us_foreign_policy', 'virology', 'world_religions']

    dataset = load_dataset("cais/mmlu", "all")
    test_data = dataset["test"]
    # Print the number of test examples

    return test_data


# Tokenizer for counting tokens
def count_tokens(text: str = None):
    tokenizer = tiktoken.encoding_for_model("gpt-4o")
    return len(tokenizer.encode(text))


# Evaluate model on MMLU
def evaluate_model_on_mmlu(agent: Agent, test_data):
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
    questions = test_data["question"]
    subjects = test_data["subject"]
    choices = test_data["choices"]
    correct_answer = test_data["answer"]
    for i in range(total):
        question = questions[i]
        subject = subjects[i]
        choice = choices[i]
        choice_str = str(choice)
        answer = correct_answer[i]
        print(f"Question {i+1}: {question}")
        print(f"  Subject: {subject}")
        print(f"  Answer: {choice}")
        print(f"  Correct Choice: {answer}: {choice[answer]}")

        start_time = time.time()
        # Call the LLM to get the answer
        predicted_answer = agent.run(question + choice_str)
        end_time = time.time()
        latency = end_time - start_time
        total_time += latency
        predicted_answer = normalize_answer(predicted_answer)
        # Compare the predicted answer with the correct answer
        if predicted_answer == choice[answer]:
            print(f"Question {i+1}: Correct")
            correct += 1
        else:
            print(f"Question {i+1}: Incorrect")
            print(f"  Question: {question}")
            print(f"  Correct Choice: {answer}: {choice[answer]}")
            print(f"  Predicted Answer: {predicted_answer}")
        # Count tokens
        tokens = count_tokens(question + choice_str + predicted_answer)
        total_tokens += tokens

        # Log metrics
        logger.info(f"Problem: {question}")
        logger.info(f"Subject: {subject}")
        logger.info(f"Choices: {choice}")
        logger.info(f"Predicted Answer: {predicted_answer}")
        logger.info(f"Correct Choice: {answer}: {choice[answer]}")
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
    agent_name="MMLU-Agent",
    system_prompt=MMLU_PROMPT,
    llm=OpenAIChat(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    ),
    max_loops=1,
    autosave=True,
    verbose=True,
)

# Load the MATH dataset
test_data = load_mmlu_dataset()

# Evaluate the agent on MATH
accuracy, avg_latency, avg_tokens = evaluate_model_on_mmlu(
    agent, test_data
)

print(f"MMLU Evaluation Accuracy: {accuracy * 100:.2f}%")
print(f"Average Latency: {avg_latency:.2f} seconds")
print(f"Average Tokens: {avg_tokens}")
