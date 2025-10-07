# Math Tutoring Evaluation Framework

This directory contains the evaluation framework for math tutoring simulations in SimulatorArena, supporting two primary use cases.

## 📁 Directory Structure

```
SimulatorArena/evaluation/math_tutoring/
├── scripts/                              # Evaluation scripts
│   ├── generate_batch_prompts_for_*.py  # Generate evaluation prompts
│   ├── process_batch_evaluation.py      # Submit/retrieve batch results
│   ├── show_assistant_performance.py    # Analyze assistant models
│   └── show_simulator_performance.py    # Analyze simulator fidelity
├── prompts/                              # Evaluation prompt templates
├── batch_prompts/                        # Generated batch requests (runtime)
├── evaluation_outputs/                   # Evaluation results (runtime)
├── evaluate_assistants.sh               # Script for assistant evaluation
└── evaluate_simulators.sh               # Script for simulator evaluation
```

**Input Data Location**: 
- Terminated conversations: `../../simulation/terminated_conversations/`
- Annotations: `../../data/math_tutoring_annotations.json`

## 🎯 Two Primary Use Cases

### Part 1: Evaluate Assistant Models (Benchmarking Mode)

Evaluate the performance of different AI assistant models as math tutors.

**Quick Start**:
```bash
# Run complete evaluation for assistant models
bash ./evaluate_assistants.sh \
    --file_name "gpt-5-mini/zero-shot-cot-user-profile_up-interaction_style_for_benchmarking" \
    --annotation_id "math_tutoring_annotations"
```

**Metrics Computed**:
- **Average Conversation Turns**: Mean number of turns until problem completion (efficiency metric)
- **Average Interaction Rating**: Mean rating on 1-10 scale using GPT-4o evaluation of conversation quality
- **Answer Correctness Rate**: Percentage of conversations where student reaches correct answer
- **Sample Count**: Number of conversations evaluated per model

### Part 2: Evaluate User Simulators

Assess how well user simulators mimic human behavior.

**Quick Start**:
```bash
# Run complete evaluation for a user simulator
bash ./evaluate_simulators.sh \
    --file_name "gpt-5-mini/zero-shot-cot" \
    --annotation_id "math_tutoring_annotations"
```

**Metrics Computed**:

**Rating Correlation Metrics** (Spearman, Pearson, Kendall correlations):
- **Instance Level**: Correlation across all conversation pairs (~450 data points)
- **Intermediate Level**: Correlation at (model, difficulty_level) granularity (~27 points for 9 models × 3 difficulty levels)
- **System Level**: Correlation at model-average level (~9 points for 9 models)

**Essence Metrics** (Answer Correctness Prediction):
- **F1 Correct**: F1 score for predicting "correct" answers (2×precision×recall/(precision+recall))
- **F1 Incorrect**: F1 score for predicting "incorrect" answers
- **Macro F1**: Average of F1 Correct and F1 Incorrect
- **Accuracy**: Overall correctness prediction accuracy ((TP+TN)/(TP+TN+FP+FN))
- **Confusion Matrix**: True/False Positives/Negatives for correctness prediction

## 📊 Detailed Evaluation Pipeline

### Step 1: Interaction Rating Evaluation

Evaluates the quality of tutor-student interactions.

```bash
# Generate batch prompts for interaction rating
python scripts/generate_batch_prompts_for_interaction_rating.py \
    --file_name "gpt-5-mini/zero-shot-cot" \
    --annotation_id "math_tutoring_annotations" \
    --terminate_help

# Submit batch to OpenAI (50% cost reduction)
python scripts/process_batch_evaluation.py \
    --batch_file "batch_prompts/interaction_rating/zero-shot-cot_v1.jsonl" \
    --model "gpt-5-mini"

# Retrieve results (after batch completes)
python scripts/process_batch_evaluation.py \
    --batch_file "batch_prompts/interaction_rating/zero-shot-cot_v1.jsonl" \
    --batch_id "batch_xxx" \
    --log_file "batch_prompts/interaction_rating/zero-shot-cot_v1_log.json"
```

### Step 2: Answer Extraction

Extracts the student's final answer from conversations.

```bash
# Generate batch prompts for answer extraction
python scripts/generate_batch_prompts_for_answer_extraction.py \
    --file_name "gpt-5-mini/zero-shot-cot" \
    --annotation_id "math_tutoring_annotations" \
    --terminate_help

# Submit batch
python scripts/process_batch_evaluation.py \
    --batch_file "batch_prompts/extracted_answer/zero-shot-cot.jsonl" \
    --model "gpt-5-mini"

# Retrieve results
python scripts/process_batch_evaluation.py \
    --batch_file "batch_prompts/extracted_answer/zero-shot-cot.jsonl" \
    --batch_id "batch_xxx" \
    --log_file "batch_prompts/extracted_answer/zero-shot-cot_log.json"
```

### Step 3: Correctness Check

Evaluates whether extracted answers are mathematically correct.

```bash
# Generate batch prompts for correctness check
python scripts/generate_batch_prompts_for_correctness_check.py \
    --file_name "gpt-5-mini/zero-shot-cot" \
    --annotation_id "math_tutoring_annotations" \
    --terminate_help

# Submit batch
python scripts/process_batch_evaluation.py \
    --batch_file "batch_prompts/extracted_answer/zero-shot-cot_correctness.jsonl" \
    --model "gpt-5-mini"

# Retrieve results
python scripts/process_batch_evaluation.py \
    --batch_file "batch_prompts/extracted_answer/zero-shot-cot_correctness.jsonl" \
    --batch_id "batch_xxx" \
    --log_file "batch_prompts/extracted_answer/zero-shot-cot_correctness_log.json"
```

## 📈 Performance Analysis

### For Assistant Models (Part 1)

Analyze the performance of different AI tutors:

```bash
# Show assistant performance table
python scripts/show_assistant_performance.py \
    --annotation_id "math_tutoring_annotations" \
    --file_name "gpt-5-mini/zero-shot-cot_benchmarking" \
    --sort_by "correctness"

# Export results in different formats
python scripts/show_assistant_performance.py \
    --file_name "gpt-5-mini/zero-shot-cot_benchmarking" \
    --output_format "latex" \
    --top_n 10
```

**Output Columns**:
- **Model**: Assistant model name
- **Avg Turns**: Mean conversation length across all problems (lower may indicate efficiency)
- **Avg Rating**: Mean GPT-4o evaluation score (1-10 scale, higher is better)
- **Correctness %**: Percentage of conversations where student reaches correct answer
- **Sample Count**: Number of evaluated conversations for statistical validity

### For User Simulators (Part 2)

Evaluate simulator fidelity against human behavior:

```bash
# Compare multiple simulators
python scripts/show_simulator_performance.py \
    --annotation_id "math_tutoring_annotations" \
    --file_names "gpt-5-mini/zero-shot" "gpt-5-mini/zero-shot-cot" "gpt-5-mini/zero-shot-cot-user-profile_up-interaction_style"

# Export for publication
python scripts/show_simulator_performance.py \
    --file_names "gpt-5-mini/zero-shot-cot" "gpt-5-mini/zero-shot-cot-user-profile_up-interaction_style" \
    --output_format "latex"
```

**Output Tables**:

**Rating Correlation Table**:
- **Simulator**: Name of the user simulator being evaluated
- **Instance Spearman/Pearson/Kendall**: Correlations across all ~450 conversation pairs
- **Intermediate Spearman/Pearson/Kendall**: Correlations at (model, difficulty) level (~27 points)
- **System Spearman/Pearson/Kendall**: Correlations at model-average level (~9 points)

**Essence (F1) Table**:
- **Simulator**: Name of the user simulator being evaluated
- **F1 Correct**: F1 score for predicting when student gets correct answer
- **F1 Incorrect**: F1 score for predicting when student gets incorrect answer
- **Macro F1**: (F1_correct + F1_incorrect) / 2
- **Accuracy**: Overall correctness prediction accuracy
- **N**: Number of evaluated instances

## 🔧 Advanced Configuration

### Batch Processing Options

The batch processing script **automatically waits** for batch completion by default and is **idempotent** (safe to rerun).

#### Idempotent Behavior (Safe Rerun)

The script **automatically checks** for existing batches before submitting duplicates:

```bash
# First run - submits batch and waits
python scripts/process_batch_evaluation.py \
    --batch_file "batch_prompts/interaction_rating/gpt-5-mini/test.json" \
    --model "gpt-5-mini"
# → Submits batch abc123, saves to logs/
# → Waits for completion

# User cancels or times out (Ctrl+C)
# Later, rerun SAME command
python scripts/process_batch_evaluation.py \
    --batch_file "batch_prompts/interaction_rating/gpt-5-mini/test.json" \
    --model "gpt-5-mini"
# → Finds existing batch abc123 in logs/
# → Checks status: "in_progress"
# → RESUMES waiting (NO duplicate batch!)
# → Retrieves results when complete

# Much later, rerun again
python scripts/process_batch_evaluation.py \
    --batch_file "batch_prompts/interaction_rating/gpt-5-mini/test.json" \
    --model "gpt-5-mini"
# → Finds existing batch abc123
# → Checks status: "completed"
# → Retrieves results immediately (NO duplicate batch!)
```

**Batch metadata** is automatically saved to `batch_prompts/{evaluation_type}/logs/{filename}.json` for tracking.

#### Advanced Options

```bash
# Customize polling interval
python scripts/process_batch_evaluation.py \
    --batch_file "path/to/batch.json" \
    --model "gpt-5-mini" \
    --polling_interval 30

# Force new batch (ignore existing)
python scripts/process_batch_evaluation.py \
    --batch_file "path/to/batch.json" \
    --model "gpt-5-mini" \
    --no_check_existing

# Check status only (no waiting)
python scripts/process_batch_evaluation.py \
    --batch_file "path/to/batch.json" \
    --batch_id "batch_xxx" \
    --check_status

# Retrieve results with specific batch ID
python scripts/process_batch_evaluation.py \
    --batch_file "path/to/batch.json" \
    --batch_id "batch_xxx" \
    --log_file "path/to/log.json"
```

**Parameters**:
- `--polling_interval`: Polling frequency in seconds (default: 60)
- `--wait`: Wait for batch to complete (automatic when no --batch_id provided)
- `--check_status`: Check status of existing batch without waiting
- `--no_check_existing`: Force new batch submission (ignore existing batches)
- `--batch_id`: Use specific batch ID (skips automatic batch check)
- `--log_file`: Path to log file for batch with --batch_id

### Custom Evaluation Parameters

**For Interaction Rating**:
- `--evaluator_model`: Model for evaluation (default: "gpt-5-mini")

**For Performance Analysis**:
- `--normalize`: Normalize ratings for correlation (default: True)
- `--desired_models`: Specific models to include in analysis
- `--output_format`: Output format (table/latex/json)

## 📝 Script Naming Convention

- `generate_batch_prompts_for_*.py` - Creates evaluation prompts
- `process_batch_evaluation.py` - Handles OpenAI batch API
- `show_*_performance.py` - Analyzes and displays results

## 🔄 Complete Workflow

### Evaluating New Assistant Models

1. Run simulations with benchmarking mode (see `../../simulation/README.md`)
2. Generate terminated conversations
3. Run `./evaluate_assistants.sh` with appropriate file name
4. Analyze results with `show_assistant_performance.py`

### Evaluating New User Simulators

1. Run simulations with your new simulator
2. Generate terminated conversations
3. Run `./evaluate_simulators.sh` with your simulator's output
4. Compare with baselines using `show_simulator_performance.py`

## 💾 Data Requirements

### Input Files
- **Simulation outputs**: `../../simulation/output/{annotation_id}/{file_name}.json`
  - Example: `../../simulation/output/math_tutoring_annotations/gpt-5-mini/zero-shot-cot.json`
- **Terminated conversations**: `../../simulation/terminated_conversations/{annotation_id}/{file_name}.json`
  - Example: `../../simulation/terminated_conversations/math_tutoring_annotations/gpt-5-mini/zero-shot-cot.json`
- **Annotations**: `../../data/math_tutoring_annotations.json`

### Output Files
- **Batch prompts**: `batch_prompts/{evaluation_type}/{file_name}.jsonl`
- **Evaluation results**: `evaluation_outputs/{evaluation_type}/{file_name}.json`

**Note**: The `{file_name}` includes the user model directory (e.g., `gpt-5-mini/zero-shot-cot`). With the default user simulator `gpt-5-mini`, file names will be like `gpt-5-mini/zero-shot-cot`.

## 🔑 Environment Setup

Required in `~/.env`:
```bash
OPENAI_API_KEY=your_openai_key
```

## 📚 Additional Resources

- **Simulation Framework**: See `../../simulation/README.md`
- **Data Format**: Check `../../data/` for annotation structure
- **Prompt Templates**: Review `prompts/` for evaluation criteria