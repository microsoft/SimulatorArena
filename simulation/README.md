# SimulatorArena User Simulation Framework

This directory contains the user simulation framework for SimulatorArena, enabling multi-turn interaction evaluation for Math Tutoring and Document Creation tasks.

## 📁 Repository Structure

```
SimulatorArena/
├── simulation/           # User simulation framework (THIS FOLDER)
│   ├── prompts/         # Prompt templates for simulators
│   ├── output/          # Simulation results (created at runtime)
│   └── terminated_conversations/  # Conversation termination analysis
├── evaluation/          # Evaluation framework for simulators and assistants
├── data/               # Annotations and user profiles
└── crowdsourcing/      # Human data collection tools
```

**Note**: This folder is for running simulations. To evaluate the performance of assistants or user simulators, see the `evaluation/` folder.

## 🎯 Two Primary Use Cases

### 1. Test New Assistant Models with Our Best User Simulators

Evaluate new AI assistant models on our multi-turn benchmark using state-of-the-art user simulators.

#### Math Tutoring - Best Simulator Configuration
```bash
python user_simulation_math_tutoring.py \
    --version=zero-shot-cot-user-profile \
    --user_profile_version=interaction_style \
    --benchmarking \
    --allowed_models "gpt-5,claude-sonnet-4-20250514,gemini-2.5-pro"
```

#### Document Creation - Best Simulator Configuration  
```bash
python user_simulation_document_creation.py \
    --version=zero-shot-cot-user-profile \
    --user_profile_version=preference_and_writing_interaction_style \
    --user_model=gemini-2.0-flash \
    --benchmarking \
    --allowed_models "gpt-5,claude-opus-4-1-20250805,gemini-2.5-flash"
```

**Benchmarking Parameters**:
- `--benchmarking`: Enable benchmarking mode to test multiple assistant models
- `--allowed_models`: Comma-separated list of assistant models to evaluate
- Output files will have `_benchmarking` suffix

### 2. Test and Develop New User Simulators

Design and evaluate improved user simulators that better mimic human behavior.

#### Running Different Simulator Configurations

We provide shell scripts with various simulator configurations:
- `user_simulation_math_tutoring.sh` - Math tutoring configurations
- `user_simulation_document_creation.sh` - Document creation configurations

These scripts demonstrate different simulator strategies from basic to advanced:
1. Basic zero-shot generation
2. Chain-of-thought reasoning (CoT)
3. Profile-based simulation with different feature combinations
4. Length control and refinement options

**Workflow for Testing New Simulators**:
1. Run your simulator on the same data used in human evaluation
2. Results are saved in `output/{annotation_id}/`
3. Run termination detection to identify conversation endpoints
4. Evaluate performance in the `evaluation/` folder
5. Compare against human ratings and baseline simulators

#### Designing Custom User Simulator Prompts

To create your own user simulator:

1. **Create prompt templates** in `prompts/{task}/`:
   - `{your-strategy}-initial-query.txt` - For generating the first user query (no conversation history)
   - `{your-strategy}.txt` - For generating subsequent queries with conversation history

2. **Example prompt structure**:
   ```
   prompts/
   ├── math_tutoring/
   │   ├── zero-shot-cot-initial-query.txt    # First query generation
   │   └── zero-shot-cot.txt                   # Subsequent queries
   └── document_creation/
       ├── zero-shot-cot-initial-query.txt
       └── zero-shot-cot.txt
   ```

3. **Run your custom simulator**:
   ```bash
   python user_simulation_math_tutoring.py --version={your-strategy}
   ```

## 🚀 Quick Start

### Prerequisites

Set up environment variables in `~/.env`:
```bash
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key  
GOOGLE_API_KEY=your_google_key
```

For running open-source models locally with vLLM (optional):
```bash
pip install vllm  # Requires CUDA-capable GPU
```

### Basic Usage Examples

#### Baseline Simulator (Chain-of-Thought)
```bash
# Math Tutoring
python user_simulation_math_tutoring.py --version=zero-shot-cot

# Document Creation
python user_simulation_document_creation.py --version=zero-shot-cot
```

#### Profile-Based Simulator
```bash
# Math Tutoring with interaction style
python user_simulation_math_tutoring.py \
    --version=zero-shot-cot-user-profile \
    --user_profile_version=interaction_style

# Document Creation with combined profiles
python user_simulation_document_creation.py \
    --version=zero-shot-cot-user-profile \
    --user_profile_version=preference_and_writing_interaction_style
```

## ⚙️ Configuration Options

### Core Parameters

| Parameter | Description | Options |
|-----------|-------------|---------|
| `--version` | Simulator strategy | `zero-shot`, `zero-shot-cot`, `zero-shot-cot-user-profile` |
| `--user_model` | Model for user simulation | `gpt-4o` (default 2024-05-13 version), `gemini-2.0-flash`, etc. |
| `--annotation_id` | Dataset identifier | `math_tutoring_annotations`, `document_creation_annotations` |
| `--benchmarking` | Enable benchmarking mode | Flag to test multiple assistants |
| `--allowed_models` | Assistant models to test | Comma-separated list (benchmarking mode) |

### Profile Options

**Math Tutoring Profiles**:
- `interaction_style` - User interaction patterns
- `knowledge_state` - Mathematical understanding level  
- `writing_style` - Communication style
- `knowledge_state_and_writing_interaction_style` - Combined profile

**Document Creation Profiles**:
- `preference` - Document preferences
- `writing_style` - Writing characteristics
- `interaction_style` - Interaction patterns
- `preference_and_writing_interaction_style` - Combined profile (best performing)

### Advanced Options

- `--length_control` - Control response length consistency
- `--refinement` - Enable iterative refinement
- `--refinement_message_style` - Style for refinement

## 🖥️ Running Open-Source Models with vLLM

SimulatorArena supports running open-source models locally using vLLM for both user simulators and assistant models.

### Setup

1. **Install vLLM** (requires CUDA GPU):
   ```bash
   pip install vllm
   ```

2. **Set model cache directory** (optional):
   ```bash
   export HF_HOME=/path/to/model/cache
   ```

### Basic Usage

```python
import asyncio
from utils import (
    initialize_vllm_model, 
    cleanup_vllm_model,
    simulate_conversation_in_batch_math_tutoring
)

async def run_with_vllm():
    # Initialize model once
    vllm_model = initialize_vllm_model(
        "meta-llama/Llama-3.1-8B-Instruct",
        max_model_len=8192,
        gpu_memory_utilization=0.8
    )
    
    # Use same model for both user and assistant
    vllm_models = {"default": vllm_model}
    
    try:
        # Run simulation
        results = await simulate_conversation_in_batch_math_tutoring(
            problems=["Solve: 2x + 5 = 13"],
            user_model_name="meta-llama/Llama-3.1-8B-Instruct",
            assistant_model_name="meta-llama/Llama-3.1-8B-Instruct",
            # ... other parameters ...
            vllm_models=vllm_models
        )
    finally:
        # Always clean up
        cleanup_vllm_model(vllm_model)

asyncio.run(run_with_vllm())
```

### Different Models for User and Assistant

```python
# Load different models
user_model = initialize_vllm_model("meta-llama/Llama-3.1-8B-Instruct")
assistant_model = initialize_vllm_model("mistralai/Mistral-7B-Instruct-v0.3")

vllm_models = {
    "meta-llama/Llama-3.1-8B-Instruct": user_model,
    "mistralai/Mistral-7B-Instruct-v0.3": assistant_model
}

# Use in simulation - models are matched by name
```

### Memory Optimization

For large models or limited GPU memory:

```python
vllm_model = initialize_vllm_model(
    "Qwen/Qwen3-8B",
    max_model_len=4096,  # Reduce context length
    gpu_memory_utilization=0.8,  # Use less GPU memory
    tensor_parallel_size=2,  # Use multiple GPUs if available
)
```

### Supported Models

vLLM supports most popular open-source models:
- **Llama**: 3.1, 3, 2 (8B, 70B, etc.)
- **Qwen**: Qwen3
- **Gemma**: Gemma3
- **Phi**: Phi-4

Check [vLLM documentation](https://docs.vllm.ai/en/latest/models/supported_models.html) for the complete list.

## 📊 Conversation Termination

After running simulations, identify natural conversation endpoints:

```bash
# Math Tutoring
python terminate_conversation_math_tutoring.py \
    --annotation_id math_tutoring_annotations \
    --simulation_path zero-shot-cot.json

# Document Creation  
python terminate_conversation_document_creation.py \
    --annotation_id document_creation_annotations \
    --simulation_path zero-shot-cot.json
```

## 📈 Performance Benchmarks

### Current Best Simulators

**Math Tutoring**:
- Baseline (CoT): ~0.61 Spearman correlation
- Best (Profile-based): ~0.77 Spearman correlation

**Document Creation**:
- Baseline (CoT): ~0.55 Spearman correlation
- Best (Profile-based): ~0.70 Spearman correlation

## 🔄 Complete Workflow

1. **Choose your use case**:
   - Testing assistant models → Use benchmarking mode with best simulators
   - Developing simulators → Create custom prompts and configurations

2. **Run simulations**:
   - Outputs saved to `output/{annotation_id}/`
   - JSON format with full conversation histories

3. **Process conversations**:
   - Run termination detection scripts
   - Results in `terminated_conversations/`

4. **Evaluate performance**:
   - Navigate to `../evaluation/` folder
   - Follow evaluation README for metrics computation
   - Compare against human ratings

## 📝 Output Format

Simulation outputs (JSON) contain:
- Complete conversation histories
- User simulator reasoning traces
- Assistant model responses
- Metadata (timestamps, models, parameters)
- Turn-by-turn interactions

## 🤝 Contributing

To contribute new simulators:
1. Design prompts following the template structure
2. Test on both Math Tutoring and Document Creation
3. Evaluate using the evaluation pipeline
4. Compare performance against baselines
5. Document your approach and results

## 📚 Additional Resources

- **Evaluation Pipeline**: See `../evaluation/README.md`
- **Data Format**: Check `../data/` for annotation structure
- **Prompt Engineering**: Examine existing prompts in `prompts/` for inspiration
- **Human Data**: See `../crowdsourcing/` for data collection methodology