# SimulatorArena

**Are User Simulators Reliable Proxies for Multi-Turn Evaluation of AI Assistants?**

## Welcome!

This repository contains the code and data for SimulatorArena, a framework that enables: (1) benchmarking AI assistants through multi-turn conversations with user simulators, and (2) evaluating the reliability of user simulators as proxies for human users.

## 🔧 Environment Setup

### Prerequisites
- Python 3.11 or higher
- API keys for the models you want to use (OpenAI, Anthropic, Google, Azure, etc.)

### Installation
```bash
# Clone the repository
git clone https://github.com/microsoft/SimulatorArena.git
cd SimulatorArena

# Install dependencies
pip install -r requirements.txt
```

### API Configuration
Create a `~/.env` file with your API keys:
```bash
# OpenAI
OPENAI_API_KEY=your_openai_key

# Anthropic (Claude)
ANTHROPIC_API_KEY=your_anthropic_key

# Google (Gemini)
GOOGLE_API_KEY=your_google_key

# Azure (optional)
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=your_azure_endpoint

# Mistral (optional)
MISTRAL_API_KEY=your_mistral_key
```

## Repository Structure

### 📁 `data/`
Contains real human–AI dialogues for two tasks:
- **Math tutoring**: 450 conversations with math problems of varying difficulty (requires MATH dataset - see [data/README.md](data/README.md))
- **Document creation**: 459 conversations for creating blog posts, emails, and creative writing

Each conversation is fully annotated with quality ratings. The directory also includes GPT-4o extracted user profiles in the `user_simulator_profiles/` subfolder based on the conversations, which can be used in the user simulator to create more realistic user behaviors.

### 📁 `simulation/`
Framework for running user simulations and evaluating AI assistants through multi-turn conversations. Supports two primary use cases:
1. **Benchmarking assistant models** - Test new AI assistants with our best user simulator
2. **Developing user simulators** - Create and evaluate new user simulation approaches

See [simulation/README.md](simulation/README.md) for detailed documentation.

### 📁 `evaluation/`
Comprehensive evaluation pipelines for analyzing simulation results:
- **Math tutoring evaluation**: Correctness rates, interaction ratings, and F1 scores
- **Document creation evaluation**: Document quality ratings, interaction ratings, and correlation metrics

See [evaluation/math_tutoring/README.md](evaluation/math_tutoring/README.md) and [evaluation/document_creation/README.md](evaluation/document_creation/README.md) for task-specific details.

### 📁 `crowdsourcing/`
Resources for human evaluation:
- Web interface code for data collection
- Heroku deployment scripts
- Amazon Mechanical Turk job launching scripts

## 🚀 Quick Start Guide

### Use Case 1: Evaluate Your AI Assistant

Want to benchmark your assistant model against others? Follow these steps:

#### Step 1: Run Simulations with Your Assistant
```bash
cd simulation

# For math tutoring
python user_simulation_math_tutoring.py \
    --benchmarking \
    --allowed_models "your-model-name,gpt-4o,claude-3-opus" \
    --num_workers 10

# For document creation
python user_simulation_document_creation.py \
    --benchmarking \
    --allowed_models "your-model-name,gpt-4o,claude-3-opus" \
    --num_workers 10
```

#### Step 2: Generate Termination Points of the Simulated Conversations
```bash
# Math tutoring
python terminate_conversation_math_tutoring.py \
    --simulation_path "output/math_tutoring_annotations/your_simulation.json"

# Document creation
python terminate_conversation_document_creation.py \
    --simulation_path "output/document_creation_annotations/your_simulation.json"
```

#### Step 3: Run Evaluation Pipeline
```bash
# Math tutoring evaluation
cd ../evaluation/math_tutoring
./evaluate_assistants.sh --file_name "your_simulation"

# Document creation evaluation
cd ../evaluation/document_creation
./evaluate_assistants.sh --file_name "your_simulation"
```

#### Step 4: View Results
```bash
# View performance metrics
python scripts/show_assistant_performance.py \
    --file_name "your_simulation" \
    --sort_by "correctness"  # For math tutoring
    # --sort_by "document"    # For document creation
```

### Use Case 2: Develop and Evaluate User Simulators

Want to test different user simulator configurations or develop your own? Follow these steps:

#### Step 1: Choose or Create Your Simulator Configuration

**Option A: Test existing configurations**
```bash
cd simulation

# Test different simulator strategies (see shell scripts for examples)
bash user_simulation_math_tutoring.sh    # Various math tutoring configs
bash user_simulation_document_creation.sh # Various document creation configs
```

**Option B: Create custom simulator prompts**
```bash
# Create your prompt templates in simulation/prompts/{task}/
# - {your-strategy}-initial-query.txt  (for first query)
# - {your-strategy}.txt                (for subsequent queries)

# Then run with your custom strategy
python user_simulation_math_tutoring.py --version="your-strategy"
```

#### Step 2: Run Simulations
```bash
# Math tutoring with specific configuration
python user_simulation_math_tutoring.py \
    --version="zero-shot-cot-user-profile" \
    --user_profile_version="interaction_style" \
    --user_model="gpt-4o-2024-11-20"

# Document creation with specific configuration  
python user_simulation_document_creation.py \
    --version="zero-shot-cot-user-profile" \
    --user_profile_version="preference_and_writing_interaction_style" \
    --user_model="gemini-2.0-flash"
```

#### Step 3: Generate Termination Points
```bash
# Math tutoring
python terminate_conversation_math_tutoring.py \
    --simulation_path "output/math_tutoring_annotations/your_simulation.json"

# Document creation
python terminate_conversation_document_creation.py \
    --simulation_path "output/document_creation_annotations/your_simulation.json"
```

#### Step 4: Evaluate Simulator Performance
```bash
# Math tutoring evaluation
cd ../evaluation/math_tutoring
./evaluate_simulators.sh --file_name "your_simulation"

# Document creation evaluation
cd ../evaluation/document_creation
./evaluate_simulators.sh --file_name "your_simulation"
```

#### Step 5: Analyze Results
```bash
# View correlation metrics
python scripts/show_simulator_performance.py \
    --file_name "your_simulation" \
    --aspect "both"  # Shows both interaction and outcome metrics
```

## 📊 Key Metrics

### For Assistant Evaluation:
- **Math Tutoring**: Correctness rate (% of problems solved correctly), interaction quality (1-10 scale), conversation efficiency (turn count)
- **Document Creation**: Document quality (1-10 scale), interaction quality (1-10 scale), conversation efficiency (turn count)

### For Simulator Evaluation:
- **Correlation Metrics**: Spearman, Pearson, and Kendall correlations at instance/intermediate/system levels
- **F1 Scores** (Math tutoring only): Correctness prediction accuracy for student answers

## 📊 Benchmark Results

### Assistant Model Performance

Here are the latest results from evaluating various AI assistants using our best user simulators:

| Model | Math Turns | Math Interaction (1–10) | Math Correct Rate (%) | Doc Turns | Doc Interaction (1–10) | Doc Rating (1–10) |
|:------|-----------:|------------------------:|----------------------:|----------:|------------------------:|------------------:|
| <img src="assets/icons/openai.png" width="16" height="16"> GPT-5 | 7.7 | 8.89 | 90.0 | 7.7 | 9.08 | 8.96 |
| <img src="assets/icons/claude.png" width="16" height="16"> Claude 3.7 Sonnet | 7.8 | 8.70 | 90.0 | 7.6 | 9.10 | 8.73 |
| <img src="assets/icons/claude.png" width="16" height="16"> Claude 4.1 Opus | 10.2 | 8.71 | 82.0 | 6.4 | 9.10 | 8.90 |
| <img src="assets/icons/openai.png" width="16" height="16"> GPT-4 Turbo | 7.5 | 8.60 | 84.0 | 8.4 | 9.04 | 8.50 |
| <img src="assets/icons/openai.png" width="16" height="16"> GPT-4o | 7.9 | 8.84 | 76.0 | 7.0 | 9.02 | 8.59 |
| <img src="assets/icons/claude.png" width="16" height="16"> Claude 4 Sonnet | 10.6 | 8.74 | 70.0 | 6.9 | 9.07 | 8.80 |
| <img src="assets/icons/openai.png" width="16" height="16"> GPT-4.1 | 10.3 | 8.87 | 76.0 | 7.5 | 9.08 | 8.47 |
| <img src="assets/icons/microsoft.png" width="16" height="16"> Phi-4 | 6.0 | 8.66 | 84.0 | 7.2 | 8.96 | 8.39 |
| <img src="assets/icons/claude.png" width="16" height="16"> Claude 3.5 Sonnet | 8.8 | 8.66 | 76.0 | 8.4 | 9.06 | 8.41 |
| <img src="assets/icons/openai.png" width="16" height="16"> GPT-4o mini | 9.3 | 8.56 | 76.0 | 9.0 | 8.98 | 7.98 |
| <img src="assets/icons/gemini.png" width="16" height="16"> Gemini 2.5 Flash | 12.3 | 8.38 | 52.0 | 7.5 | 9.04 | 8.70 |
| <img src="assets/icons/gemini.png" width="16" height="16"> Gemini 2.5 Pro | 13.0 | 8.36 | 48.0 | 6.3 | 9.02 | 8.66 |
| <img src="assets/icons/gemini.png" width="16" height="16"> Gemini 2.0 Flash | 11.8 | 8.36 | 58.0 | 7.7 | 8.94 | 8.36 |
| <img src="assets/icons/mistral.png" width="16" height="16"> Mistral Large 2 | 10.0 | 8.08 | 64.0 | 7.8 | 8.98 | 8.25 |
| <img src="assets/icons/llama.png" width="16" height="16"> Llama 3.3 70B | 8.2 | 8.26 | 68.0 | 8.2 | 8.88 | 7.92 |
| <img src="assets/icons/llama.png" width="16" height="16"> Llama 3.1 70B | 8.7 | 7.70 | 70.0 | 8.8 | 8.86 | 8.00 |
| <img src="assets/icons/llama.png" width="16" height="16"> Llama 3.1 8B | 10.8 | 6.48 | 46.0 | 8.8 | 8.82 | 7.53 |
| <img src="assets/icons/microsoft.png" width="16" height="16"> Phi-3 Medium | 6.9 | 6.35 | 51.0 | 9.5 | 5.57 | 7.50 |

**Note:** Models are sorted by the mean z-score across all four metrics (interaction ratings and outcomes for both tasks). All models were evaluated in non-thinking mode. For GPT-5, we set the reasoning effort to minimal, and for Gemini 2.5 Pro, we used a thinking budget of 128 (the minimum allowed). OpenAI's reasoning models have their temperature fixed at 1.0 since they don't support temperature changes, while all other models were evaluated with temperature = 0.

### User Simulator Performance

The following figure shows the performance of different user simulator configurations across both tasks:

<img src="assets/user_simulator_evaluation.png" width="100%" alt="User Simulator Performance Comparison">

The chart compares various simulator strategies, from basic zero-shot generation to advanced profile-based approaches with different feature combinations. Higher correlation values indicate better alignment with human behavior.


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
