# Evals to make

## Math
[ ] GSM8K

[ ] MATH

## code
[ ] HumanEval

[ ] MBPP 

[ ] Natural2Code

## commonsense reasoning
[ ] ARC (AI2 Reasoning Challenge)

[ ] HellaSwag

[ ] Winogrande

[ ] PIQA

[ ] SIQA

[ ] OpenbookQA

[ ] CommonsenseQA

## world knowledge
[ ] WebQuestions

[ ] NaturalQuestions

[ ] TriviaQA

[ ] ComplexWebQuestions

[ ] WebQuestionsSP
[ ] SearchQA
[ ] HotpotQA
[ ] DROP
[ ] WikiHop
[ ] QAngaroo
[ ] BoolQ
[ ] Multi

## reading comprehension

[ ] BoolQ

[ ] QuAC

[ ] DROP

## aggregated

[ ] MMLU

[ ] HELM

[ ] BBH

[ ] AGI Eval 

# Metrics

[ ] Task accomplished y/n

[ ] Task quality 1-5

[ ] was a tool used y/n
   [ ] which tool
   [ ] how was it used
   [ ] were tools called in the right order
   [ ] were the tools used correctly? Did the environment change?
   [ ] Did the agent output match the expected reference output?

[ ] Compute load

[ ] Cost $
   " the cost per 1000 function callings ![alt text](image.png)"

[ ] Average Latency (s)
    "Latency: we measure the latency by timing each request to the endpoint ignoring the function document preprocessing time."

[ ] Model size needed (Small, Medium, Large)
[ ] Model type needed (LM, VLM, predictive, point cloud, NeRF/Splat)
[ ] Number of trials to acceptable performance
[ ] Number of trials to best performance


[1]Functional Benchmarks for Robust Evaluation of Reasoning Performance, and the Reasoning Gap (https://arxiv.org/html/2402.19450v1) Code (https://github.com/consequentai/fneval/)

[2]explanation of metrics (https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html)

[3] tool use benchmarks (https://langchain-ai.github.io/langchain-benchmarks/notebooks/tool_usage/intro.html?ref=blog.langchain.dev) (https://blog.langchain.dev/benchmarking-agent-tool-use/)