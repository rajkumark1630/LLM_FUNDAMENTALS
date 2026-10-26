Week 4 – LLM Fundamentals Weekly Assessment
Project Overview
Duration: 1 Day (Approximately 6 Hours) Mode: Individual Hands-on Assessment
This project explores Large Language Models (LLMs), tokenization, cost estimation, model capabilities and limitations, Transformer architecture, and prompt engineering techniques.
The objective is to understand what LLMs can and cannot do before implementing them in real-world business applications.
Learning Objectives
Understand foundation models and major LLM families.
Analyze tokens, context windows, and API pricing.
Evaluate LLM capabilities and limitations.
Implement self-attention and multi-head attention using NumPy.
Visualize sinusoidal positional encoding.
Compare encoder-only and decoder-only Transformer models.
Build and evaluate a prompt engineering toolkit.
Extract structured information and validate JSON outputs.
Understand hallucination, reasoning errors, and responsible LLM usage.
Project Structure
LLM_FUNDAMENTALS/
│
├── task1_models_tokens_cost/
│   ├── cost_calculator.py
│   └── token_analysis.py
│
├── task2_capabilities_limitations/
│   ├── capabilities_lab.py
│   └── results.csv
│
├── task3_transformer_concepts/
│   ├── attention.py
│   ├── hf_examples.py
│   ├── model_notes.md
│   ├── positional_encoding.py
│   └── positional_encoding.png
│
├── task4_prompt_toolkit/
│   ├── data/
│   ├── prompts/
│   ├── results/
│   ├── local_classification.py
│   ├── local_evaluation.py
│   ├── local_resume_extraction.py
│   └── task4_prompt_toolkit.py
│
├── requirements.txt
├── .gitignore
└── README.md
Note: The project structure lists the main files. Actual contents may vary depending on the implementation.
Task 1 – Models, Tokens and Cost
Objective
Compare major LLM families and analyze tokenization, context windows, and estimated API costs.
Activities
Compared GPT, Claude, Llama, and Gemini model families.
Documented model owners, weight availability, context windows, and strengths.
Used tiktoken to analyze five text types:  
Sentence
Paragraph
Page
JSON
Tamil or Hindi sentence
Calculated words per token.
Implemented a token-based cost estimation function.
Estimated monthly cost for 1,000 chats per day, with 1,500 input tokens and 400 output tokens per chat.
Compared text sizes against 8K and 128K context windows.
Recommended models for a public customer-support bot and a private on-premises tool.
Cost Estimation
Daily usage:
Chats: 1,000
Input tokens per chat: 1,500
Output tokens per chat: 400
Daily token usage:
Input: 1,500,000 tokens
Output: 400,000 tokens
Monthly token usage (30 days):
Input: 45,000,000 tokens
Output: 12,000,000 tokens
Estimated monthly cost:
(45 × input price per million tokens) + (12 × output price per million tokens)
Actual cost depends on the selected model’s pricing and any additional charges.
Task 2 – Capabilities and Limitations Lab
Objective
Evaluate LLM performance across common tasks and identify limitations.
Activities
Tested summarization.
Tested text classification.
Tested information extraction.
Tested translation.
Tested code generation.
Rated outputs on a scale of 1–5.
Investigated knowledge-cutoff limitations using a recent event.
Compared hallucination behavior using a fictional paper or person.
Tested whether a model could retrieve a fact embedded in a long text.
Verified a multi-step arithmetic problem and a logic puzzle.
Documented five safeguards for business applications.
Business Safeguards
Validate model outputs before using them in business workflows.
Require reliable sources or retrieval for factual answers.
Use structured output schemas and input validation.
Apply access controls and protect confidential information.
Include human review for high-impact or uncertain decisions.
Task 3 – Transformer Concepts in Code
Objective
Implement and visualize fundamental Transformer concepts.
Activities
Implemented single-head self-attention using NumPy.
Calculated query, key, and value matrices.
Computed scaled dot-product attention scores.
Applied row-wise softmax.
Printed the 4 × 4 attention matrix and verified that each row sums to 1.
Extended the implementation to two attention heads.
Concatenated the heads and applied an output projection.
Implemented sinusoidal positional encoding for 10 positions and 8 dimensions.
Plotted positional encoding.
Demonstrated BERT fill-mask and GPT-2 text generation using Hugging Face.
Documented encoder-only versus decoder-only architectures and the advantages of Transformers over RNNs.
Key Concepts
Self-attention
Scaled dot-product attention
Multi-head attention
Positional encoding
Encoder-only models
Decoder-only models
Transformer architecture
Task 4 – Prompt Engineering Toolkit
Objective
Build and evaluate prompts for customer-support ticket classification and resume information extraction.
Activities
Created a golden set of 15 customer-support tickets.
Classified tickets into Billing, Technical, and Account categories with priorities.
Created five resume paragraphs with expected JSON outputs.
Developed three classification prompt versions:  
V1: Basic prompt
V2: Role, Task, Constraints, and Output format
V3: Few-shot prompt with three examples
Evaluated classification accuracy at temperature 0.
Tested five word problems with and without step-by-step instructions.
Compared creative prompt outputs at temperatures 0, 0.7, and 1.2.
Compared top-p sampling values of 0.3 and 0.95.
Extracted name, skills, and years of experience from five resumes.
Compared ordinary text output with JSON-constrained output.
Used Pydantic to validate extracted information.
Counted invalid JSON outputs.
Rewrote three deliberately poor prompts and evaluated the improvements.
Evaluation Metrics
Classification accuracy
Invalid JSON count
Output consistency
Structured data validity
Prompt quality and clarity
Important: Report actual measured accuracy and invalid-output counts from the experiment. Example values in the assessment specification are reference targets, not guaranteed results.
Technologies Used
Python
NumPy
Pandas
Matplotlib
tiktoken
Hugging Face Transformers
scikit-learn (where applicable)
Pydantic
Git and GitHub
Installation
Clone the repository:
git clone <YOUR_REPOSITORY_URL>
cd LLM_FUNDAMENTALS
Create a virtual environment:
python -m venv .venv
Activate it on Windows PowerShell:
.\.venv\Scripts\Activate.ps1
Install dependencies:
pip install -r requirements.txt
Run individual scripts from the project root. For example:
python task1_models_tokens_cost/cost_calculator.py
python task3_transformer_concepts/attention.py
Other scripts may require model downloads, API credentials, or additional setup depending on their implementation.
Security
Never commit .env files, API keys, passwords, or other secrets.
Keep .venv/ and __pycache__/ out of version control.
Store credentials in environment variables.
Use local models when appropriate for private or sensitive data.
Key Learnings
This assessment demonstrates how LLMs process text, how token usage affects cost, and why model outputs require evaluation and validation.
It also provides hands-on experience with Transformer attention mechanisms, positional encoding, prompt design, sampling parameters, structured extraction, and safeguards for business applications.
Conclusion
The Week 4 LLM Fundamentals assessment combines theory with practical experiments across four tasks. It provides a foundation for evaluating LLMs and designing safer, more reliable AI-powered applications.
References
OpenAI: https://platform.openai.com/docs
Anthropic: https://docs.anthropic.com/
Google Gemini: https://ai.google.dev/gemini-api/docs
Meta Llama: https://www.llama.com/
Hugging Face Transformers: https://huggingface.co/docs/transformers/
tiktoken: https://github.com/openai/tiktoken
NumPy: https://numpy.org/doc/
Pydantic: https://docs.pydantic.dev/
