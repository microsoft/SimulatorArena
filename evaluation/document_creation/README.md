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
./evaluate_assistants.sh \
    --file_name "zero-shot-cot-user-profile-up-preference_and_writing_interaction_style-gemini-2.0-flash" \
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
./evaluate_simulators.sh \
    --file_name "zero-shot-cot-user-profile-up-preference_and_writing_interaction_style" \
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
    --file_name "zero-shot-cot" \
    --annotation_id "document_creation_annotations" \
    --terminate_help

# Submit batch to OpenAI (50% cost reduction with batch API)
python scripts/process_batch_evaluation.py \
    --batch_file "batch_prompts/extracted_document/zero-shot-cot.jsonl" \
    --model "gpt-4o-2024-11-20" \
    --wait

# Results saved to: evaluation_outputs/extracted_document/zero-shot-cot.json
```

### Step 2: Document Quality Rating

Evaluates the quality of extracted documents based on document type and intent.

```bash
# Generate batch prompts for document rating
python scripts/generate_batch_prompts_for_rating.py \
    --file_name "zero-shot-cot" \
    --annotation_id "document_creation_annotations" \
    --aspect "document"

# Submit batch
python scripts/process_batch_evaluation.py \
    --batch_file "batch_prompts/document_rating/zero-shot-cot.jsonl" \
    --model "gpt-4o-2024-11-20" \
    --wait

# Results saved to: evaluation_outputs/document_rating/zero-shot-cot.json
```

### Step 3: Interaction Quality Rating

Evaluates the quality of assistant-user interactions during document creation.

```bash
# Generate batch prompts for interaction rating
python scripts/generate_batch_prompts_for_rating.py \
    --file_name "zero-shot-cot" \
    --annotation_id "document_creation_annotations" \
    --aspect "interaction"

# Submit batch
python scripts/process_batch_evaluation.py \
    --batch_file "batch_prompts/interaction_rating/zero-shot-cot.jsonl" \
    --model "gpt-4o-2024-11-20" \
    --wait

# Results saved to: evaluation_outputs/interaction_rating/zero-shot-cot.json
```

## 📈 Performance Analysis

### For Assistant Models (Part 1)

Analyze the performance of different AI assistants:

```bash
# Show assistant performance sorted by document quality
python scripts/show_assistant_performance.py \
    --file_name "zero-shot-cot-benchmarking" \
    --annotation_id "document_creation_annotations" \
    --sort_by "document"

# Sort by interaction quality
python scripts/show_assistant_performance.py \
    --file_name "zero-shot-cot-benchmarking" \
    --sort_by "interaction"

# Sort by combined score
python scripts/show_assistant_performance.py \
    --file_name "zero-shot-cot-benchmarking" \
    --sort_by "combined" \
    --top_k 10

# Export results
python scripts/show_assistant_performance.py \
    --file_name "zero-shot-cot-benchmarking" \
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
    --file_name "zero-shot-cot" \
    --annotation_id "document_creation_annotations" \
    --aspect "both" \
    --normalize

# Document aspect only
python scripts/show_simulator_performance.py \
    --file_name "zero-shot-cot" \
    --aspect "document" \
    --normalize

# Without normalization
python scripts/show_simulator_performance.py \
    --file_name "zero-shot-cot" \
    --aspect "both" \
    --no_normalize

# Export results
python scripts/show_simulator_performance.py \
    --file_name "zero-shot-cot" \
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

```bash
# Process with custom model
python scripts/process_batch_evaluation.py \
    --batch_file "path/to/batch.jsonl" \
    --model "gpt-4o-2024-11-20" \
    --poll_interval 60

# Check batch status
python scripts/process_batch_evaluation.py \
    --batch_file "path/to/batch.jsonl" \
    --batch_id "batch_xxx" \
    --check_status
```

### Custom Evaluation Parameters

**For Document Extraction**:
- `--terminate_help`: Use terminated conversation endpoints (highly recommended)
- `--max_tokens`: Maximum tokens for extracted document (default: 2000)

**For Rating Evaluation**:
- `--aspect`: "document" or "interaction"
- `--evaluator_model`: Model for evaluation (default: "gpt-4o-2024-11-20")

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
- **Terminated conversations**: `../../simulation/terminated_conversations/{annotation_id}/{file_name}.json`
- **Human annotations**: `../../simulation/data/document_creation_annotations.json`

### Output Files
- **Extracted documents**: `evaluation_outputs/extracted_document/*.json`
- **Document ratings**: `evaluation_outputs/document_rating/*.json`
- **Interaction ratings**: `evaluation_outputs/interaction_rating/*.json`
- **Performance summaries**: `evaluation_outputs/*_performance_*.json`


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