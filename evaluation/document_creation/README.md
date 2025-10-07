# Document Creation Evaluation Framework

This directory contains the evaluation framework for document creation simulations in SimulatorArena, supporting two primary use cases.

## 📁 Directory Structure

```
SimulatorArena/evaluation/document_creation/
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
- Simulation outputs: `../../simulation/output/`
- Terminated conversations: `../../simulation/terminated_conversations/`
- Annotations: `../../simulation/data/document_creation_annotations.json`

## 🎯 Two Primary Use Cases

### Part 1: Evaluate Assistant Models (Benchmarking Mode)

Evaluate the performance of different AI assistant models for document creation tasks.

**Quick Start**:
```bash
# Run complete evaluation for assistant models
bash ./evaluate_assistants.sh \
    --file_name "gpt-5-mini/zero-shot-cot-user-profile_up-preference_and_writing_interaction_style_for_benchmarking" \
    --annotation_id "document_creation_annotations"
```

**Metrics Computed**:
- **Average Document Rating**: Mean quality rating of final documents (1-10 scale)
- **Average Interaction Rating**: Mean quality rating of conversations (1-10 scale)
- **Average Conversation Turns**: Mean number of turns until task completion
- **Sample Count**: Number of conversations evaluated per model

### Part 2: Evaluate User Simulators

Assess how well user simulators mimic human behavior in document creation tasks.

**Quick Start**:
```bash
# Run complete evaluation for a user simulator
bash ./evaluate_simulators.sh \
    --file_name "gpt-5-mini/zero-shot-cot-user-profile_up-preference_and_writing_interaction_style" \
    --annotation_id "document_creation_annotations"
```

**Metrics Computed**:

**Correlation Metrics** (Spearman ρ, Pearson r, Kendall τ) for both document and interaction aspects:
- **Instance Level**: Correlation across all individual conversations (~450 data points)
- **Intermediate Level**: Correlation at (model, document_type) granularity (~27 points for 9 models × 3 document types)
- **System Level**: Correlation at model-average level (~9 points for 9 models)

## 📊 Detailed Evaluation Pipeline

### ⚠️ CRITICAL: Pipeline Order

**Document extraction MUST happen BEFORE document rating** because rating prompts need the extracted document content to evaluate quality.

### Step 1: Document Extraction (MUST BE FIRST!)

Extracts the final document from each conversation.

```bash
# Generate batch prompts for document extraction
python scripts/generate_batch_prompts_for_document_extraction.py \
    --file_name "gpt-5-mini/zero-shot-cot" \
    --annotation_id "document_creation_annotations" \
    --terminate_help

# Submit batch to OpenAI (50% cost reduction with batch API)
# The script will automatically wait for completion (default: 2 hours max)
python scripts/process_batch_evaluation.py \
    --batch_file "batch_prompts/extracted_document/gpt-5-mini/zero-shot-cot.jsonl" \
    --model "gpt-5-mini"

# Results saved to: evaluation_outputs/extracted_document/gpt-5-mini/zero-shot-cot.json
```

### Step 2: Document Quality Rating

Evaluates the quality of extracted documents based on document type and intent.

```bash
# Generate batch prompts for document rating
python scripts/generate_batch_prompts_for_rating.py \
    --file_name "gpt-5-mini/zero-shot-cot" \
    --annotation_id "document_creation_annotations" \
    --aspect "document"

# Submit batch
python scripts/process_batch_evaluation.py \
    --batch_file "batch_prompts/document_rating/gpt-5-mini/zero-shot-cot.jsonl" \
    --model "gpt-5-mini"

# Results saved to: evaluation_outputs/document_rating/gpt-5-mini/zero-shot-cot.json
```

### Step 3: Interaction Quality Rating

Evaluates the quality of assistant-user interactions during document creation.

```bash
# Generate batch prompts for interaction rating
python scripts/generate_batch_prompts_for_rating.py \
    --file_name "gpt-5-mini/zero-shot-cot" \
    --annotation_id "document_creation_annotations" \
    --aspect "interaction"

# Submit batch
python scripts/process_batch_evaluation.py \
    --batch_file "batch_prompts/interaction_rating/gpt-5-mini/zero-shot-cot.jsonl" \
    --model "gpt-5-mini"

# Results saved to: evaluation_outputs/interaction_rating/gpt-5-mini/zero-shot-cot.json
```

## 📈 Performance Analysis

### For Assistant Models (Part 1)

Analyze the performance of different AI assistants:

```bash
# Show assistant performance sorted by document quality
python scripts/show_assistant_performance.py \
    --file_name "gpt-5-mini/zero-shot-cot_for_benchmarking" \
    --annotation_id "document_creation_annotations" \
    --sort_by "document"

# Sort by interaction quality
python scripts/show_assistant_performance.py \
    --file_name "gpt-5-mini/zero-shot-cot_for_benchmarking" \
    --sort_by "interaction"

# Sort by combined score
python scripts/show_assistant_performance.py \
    --file_name "gpt-5-mini/zero-shot-cot_for_benchmarking" \
    --sort_by "combined" \
    --top_k 10

# Export results
python scripts/show_assistant_performance.py \
    --file_name "gpt-5-mini/zero-shot-cot_for_benchmarking" \
    --sort_by "combined" \
    --export "results/assistant_performance.json"
```

**Output Columns**:
- **Rank**: Ranking based on sort criterion
- **Model**: Assistant model name
- **Doc Rating**: Mean document quality rating (1-10, higher is better)
- **Doc Std**: Standard deviation of document ratings
- **Int Rating**: Mean interaction quality rating (1-10, higher is better)
- **Int Std**: Standard deviation of interaction ratings
- **Avg Turns**: Mean conversation length (efficiency metric)
- **N**: Number of evaluated conversations

### For User Simulators (Part 2)

Evaluate simulator fidelity against human behavior:

```bash
# Analyze both aspects with normalization
python scripts/show_simulator_performance.py \
    --file_name "gpt-5-mini/zero-shot-cot" \
    --annotation_id "document_creation_annotations" \
    --aspect "both" \
    --normalize

# Document aspect only
python scripts/show_simulator_performance.py \
    --file_name "gpt-5-mini/zero-shot-cot" \
    --aspect "document" \
    --normalize

# Without normalization
python scripts/show_simulator_performance.py \
    --file_name "gpt-5-mini/zero-shot-cot" \
    --aspect "both" \
    --no_normalize

# Export results
python scripts/show_simulator_performance.py \
    --file_name "gpt-5-mini/zero-shot-cot" \
    --aspect "both" \
    --export "results/simulator_performance.json"
```

**Output Tables**:

**Correlation Metrics for Each Aspect**:
- **Instance Level**: Individual conversation correlations
  - Spearman ρ: Rank correlation (robust to outliers)
  - Pearson r: Linear correlation
  - Kendall τ: Rank correlation (handles ties well)
- **Intermediate Level**: (Model, Document Type) aggregated correlations
- **System Level**: Model-level aggregated correlations

## 🔧 Advanced Configuration

### Batch Processing Options

The batch processing script **automatically waits** for batch completion by default and is **idempotent** (safe to rerun).

#### Idempotent Behavior (Safe Rerun)

The script **automatically checks** for existing batches before submitting duplicates:

```bash
# First run - submits batch and waits
python scripts/process_batch_evaluation.py \
    --batch_file "batch_prompts/extracted_document/gpt-5-mini/test.jsonl" \
    --model "gpt-5-mini"
# → Submits batch abc123, saves metadata to logs/
# → Waits for completion

# User cancels or times out (Ctrl+C)
# Later, rerun SAME command
python scripts/process_batch_evaluation.py \
    --batch_file "batch_prompts/extracted_document/gpt-5-mini/test.jsonl" \
    --model "gpt-5-mini"
# → Finds existing batch abc123 in logs/
# → Checks status: "in_progress"
# → RESUMES waiting (NO duplicate batch!)
# → Retrieves results when complete

# Much later, rerun again
python scripts/process_batch_evaluation.py \
    --batch_file "batch_prompts/extracted_document/gpt-5-mini/test.jsonl" \
    --model "gpt-5-mini"
# → Finds existing batch abc123
# → Checks status: "completed"
# → Retrieves results immediately (NO duplicate batch!)
```

**Batch metadata** is automatically saved to `batch_prompts/{evaluation_type}/logs/{filename}.json` for tracking.

#### Advanced Options

```bash
# Customize polling and wait time
python scripts/process_batch_evaluation.py \
    --batch_file "path/to/batch.jsonl" \
    --model "gpt-5-mini" \
    --poll_interval 30 \
    --max_wait 3600

# Force new batch (ignore existing)
python scripts/process_batch_evaluation.py \
    --batch_file "path/to/batch.jsonl" \
    --model "gpt-5-mini" \
    --no_check_existing

# Check status only (no waiting)
python scripts/process_batch_evaluation.py \
    --batch_file "path/to/batch.jsonl" \
    --batch_id "batch_xxx" \
    --check_status

# Use specific batch ID
python scripts/process_batch_evaluation.py \
    --batch_file "path/to/batch.jsonl" \
    --batch_id "batch_xxx"
```

**Parameters**:
- `--poll_interval`: Polling frequency in seconds (default: 30)
- `--max_wait`: Maximum wait time in seconds (default: 7200 = 2 hours)
- `--check_status`: Check status of existing batch without waiting
- `--no_check_existing`: Force new batch submission (ignore existing batches)
- `--batch_id`: Use specific batch ID instead of checking for existing

### Custom Evaluation Parameters

**For Document Extraction**:
- `--terminate_help`: Use terminated conversation endpoints (highly recommended)
- `--max_tokens`: Maximum tokens for extracted document (default: 5000)

**For Rating Evaluation**:
- `--aspect`: "document" or "interaction"
- `--evaluator_model`: Model for evaluation (default: "gpt-5-mini")
- `--max_tokens`: Maximum tokens for rating response (default: 5000)

**For Performance Analysis**:
- `--normalize`: Normalize human ratings per annotator (default: True)
- `--sort_by`: Sort criterion for assistants (document/interaction/combined)
- `--top_k`: Show only top k models

## 📝 Document Types and Intents

The framework evaluates three document types with various intents:

**Document Types**:
- **Blog Post**: Professional or casual blog articles
- **Email/Letter**: Formal or informal correspondence
- **Creative Writing**: Stories, poems, or creative content

**Example Intents**:
- Request information
- Express gratitude
- Persuade/argue
- Tell a story
- Provide instructions


## 💾 Data Requirements

### Input Files
- **Simulation outputs**: `../../simulation/output/{annotation_id}/{file_name}.json`
  - Example: `../../simulation/output/document_creation_annotations/gpt-5-mini/zero-shot-cot.json`
- **Terminated conversations**: `../../simulation/terminated_conversations/{annotation_id}/{file_name}.json`
  - Example: `../../simulation/terminated_conversations/document_creation_annotations/gpt-5-mini/zero-shot-cot.json`
- **Human annotations**: `../../data/document_creation_annotations.json`

### Output Files
- **Extracted documents**: `evaluation_outputs/extracted_document/{file_name}.json`
- **Document ratings**: `evaluation_outputs/document_rating/{file_name}.json`
- **Interaction ratings**: `evaluation_outputs/interaction_rating/{file_name}.json`
- **Performance summaries**: `evaluation_outputs/*_performance_*.json`

**Note**: The `{file_name}` includes the user model directory (e.g., `gpt-5-mini/zero-shot-cot`). With the default user simulator `gpt-5-mini`, file names will be like `gpt-5-mini/zero-shot-cot`.


## 📊 Key Differences from Math Tutoring

| Aspect | Document Creation | Math Tutoring |
|--------|------------------|---------------|
| **End Outcome** | Document quality rating (1-10) | Answer correctness (correct/incorrect) |
| **Interaction Metric** | Interaction quality rating (1-10) | Interaction quality rating (1-10) |
| **Pipeline Order** | Extract → Rate Document → Rate Interaction | Rate Interaction → Extract Answer → Check Correctness |
| **Simulator Metrics** | Correlations only | Correlations + F1 scores |
| **Document Types** | Blog/Email/Creative | N/A |
| **Problem Difficulty** | N/A | Easy (3)/Medium (4)/Hard (5) |


## 📚 Additional Resources

- **Simulation Framework**: See `../../simulation/README.md`
- **Data Format**: Check `../../simulation/data/` for annotation structure
- **Prompt Templates**: Review `prompts/` for evaluation criteria