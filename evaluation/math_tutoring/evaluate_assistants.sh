#!/bin/bash

# ============================================================================
# Assistant Model Evaluation Script for Math Tutoring
# ============================================================================
# This script evaluates the performance of AI assistant models as math tutors
# Used when testing new assistant models with benchmarking mode
# ============================================================================

set -e  # Exit on error

# Default values
FILE_NAME=""
ANNOTATION_ID="math_tutoring_annotations"
EVALUATOR_MODEL="gpt-4o-2024-11-20"
PROMPT_VERSION="v1"
SKIP_COMPLETED=true
VERBOSE=false

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --file_name)
            FILE_NAME="$2"
            shift 2
            ;;
        --annotation_id)
            ANNOTATION_ID="$2"
            shift 2
            ;;
        --evaluator_model)
            EVALUATOR_MODEL="$2"
            shift 2
            ;;
        --prompt_version)
            PROMPT_VERSION="$2"
            shift 2
            ;;
        --no-skip)
            SKIP_COMPLETED=false
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            echo "Usage: $0 --file_name <name> [options]"
            echo ""
            echo "Options:"
            echo "  --file_name          Name of simulation output file (required)"
            echo "  --annotation_id      Annotation dataset ID (default: math_tutoring_annotations)"
            echo "  --evaluator_model    Model for evaluation (default: gpt-4o-2024-11-20)"
            echo "  --prompt_version     Prompt version (default: v1)"
            echo "  --no-skip           Don't skip already completed evaluations"
            echo "  --verbose           Show detailed output"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$FILE_NAME" ]; then
    echo -e "${RED}Error: --file_name is required${NC}"
    echo "Use --help for usage information"
    exit 1
fi

# Function to print step headers
print_step() {
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

# Function to check if file exists
check_file() {
    if [ ! -f "$1" ]; then
        echo -e "${RED}Error: File not found: $1${NC}"
        return 1
    fi
    return 0
}

# Main evaluation pipeline
main() {
    print_step "EVALUATING ASSISTANT MODELS: $FILE_NAME"
    
    # Check for terminated conversations
    TERMINATED_FILE="../../simulation/terminated_conversations/${ANNOTATION_ID}/${FILE_NAME}.json"
    if ! check_file "$TERMINATED_FILE"; then
        echo -e "${YELLOW}Warning: Terminated conversations not found${NC}"
        echo "Please run termination detection first:"
        echo "  cd ../../simulation"
        echo "  python terminate_conversation_math_tutoring.py --annotation_id $ANNOTATION_ID --simulation_path output/${ANNOTATION_ID}/${FILE_NAME}.json"
        exit 1
    fi
    
    # ============================================================
    # STEP 1: Interaction Rating
    # ============================================================
    print_step "STEP 1: Interaction Rating Evaluation"
    
    RATING_OUTPUT="evaluation_outputs/interaction_rating/${FILE_NAME}_${PROMPT_VERSION}.json"
    
    if [ -f "$RATING_OUTPUT" ] && [ "$SKIP_COMPLETED" = true ]; then
        echo -e "${YELLOW}Skipping: Interaction rating already completed${NC}"
    else
        echo "Generating batch prompts for interaction rating..."
        python scripts/generate_batch_prompts_for_interaction_rating.py \
            --file_name "$FILE_NAME" \
            --annotation_id "$ANNOTATION_ID" \
            --terminate_help \
            --prompt_version "$PROMPT_VERSION"
        
        BATCH_FILE="batch_prompts/interaction_rating/${FILE_NAME}_${PROMPT_VERSION}.jsonl"
        
        echo "Submitting batch to OpenAI..."
        BATCH_OUTPUT=$(python scripts/process_batch_evaluation.py \
            --batch_file "$BATCH_FILE" \
            --model "$EVALUATOR_MODEL" 2>&1)
        
        # Extract batch ID
        BATCH_ID=$(echo "$BATCH_OUTPUT" | grep -oP 'batch_[a-zA-Z0-9]+' | head -1)
        
        if [ -z "$BATCH_ID" ]; then
            echo -e "${RED}Error: Failed to submit batch${NC}"
            echo "$BATCH_OUTPUT"
            exit 1
        fi
        
        echo -e "${GREEN}Batch submitted: $BATCH_ID${NC}"
        echo "Waiting for batch to complete (this may take 10-30 minutes)..."
        
        # Poll for completion
        while true; do
            STATUS=$(python scripts/process_batch_evaluation.py \
                --batch_file "$BATCH_FILE" \
                --batch_id "$BATCH_ID" \
                --check_status 2>&1 | grep -oP '(?<=Status: )[a-z]+' || echo "unknown")
            
            if [ "$STATUS" = "completed" ]; then
                break
            elif [ "$STATUS" = "failed" ] || [ "$STATUS" = "expired" ]; then
                echo -e "${RED}Batch failed with status: $STATUS${NC}"
                exit 1
            fi
            
            echo "Status: $STATUS - waiting 30 seconds..."
            sleep 30
        done
        
        echo "Retrieving results..."
        python scripts/process_batch_evaluation.py \
            --batch_file "$BATCH_FILE" \
            --batch_id "$BATCH_ID" \
            --log_file "batch_prompts/interaction_rating/${FILE_NAME}_${PROMPT_VERSION}_log.json"
        
        echo -e "${GREEN}✓ Interaction rating completed${NC}"
    fi
    
    # ============================================================
    # STEP 2: Answer Extraction
    # ============================================================
    print_step "STEP 2: Answer Extraction"
    
    ANSWER_OUTPUT="evaluation_outputs/extracted_answer/${FILE_NAME}.json"
    
    if [ -f "$ANSWER_OUTPUT" ] && [ "$SKIP_COMPLETED" = true ]; then
        echo -e "${YELLOW}Skipping: Answer extraction already completed${NC}"
    else
        echo "Generating batch prompts for answer extraction..."
        python scripts/generate_batch_prompts_for_answer_extraction.py \
            --file_name "$FILE_NAME" \
            --annotation_id "$ANNOTATION_ID" \
            --terminate_help
        
        BATCH_FILE="batch_prompts/extracted_answer/${FILE_NAME}.jsonl"
        
        echo "Submitting batch to OpenAI..."
        BATCH_OUTPUT=$(python scripts/process_batch_evaluation.py \
            --batch_file "$BATCH_FILE" \
            --model "$EVALUATOR_MODEL" 2>&1)
        
        BATCH_ID=$(echo "$BATCH_OUTPUT" | grep -oP 'batch_[a-zA-Z0-9]+' | head -1)
        
        if [ -z "$BATCH_ID" ]; then
            echo -e "${RED}Error: Failed to submit batch${NC}"
            exit 1
        fi
        
        echo -e "${GREEN}Batch submitted: $BATCH_ID${NC}"
        echo "Waiting for batch to complete..."
        
        # Poll for completion
        while true; do
            STATUS=$(python scripts/process_batch_evaluation.py \
                --batch_file "$BATCH_FILE" \
                --batch_id "$BATCH_ID" \
                --check_status 2>&1 | grep -oP '(?<=Status: )[a-z]+' || echo "unknown")
            
            if [ "$STATUS" = "completed" ]; then
                break
            elif [ "$STATUS" = "failed" ] || [ "$STATUS" = "expired" ]; then
                echo -e "${RED}Batch failed with status: $STATUS${NC}"
                exit 1
            fi
            
            echo "Status: $STATUS - waiting 30 seconds..."
            sleep 30
        done
        
        echo "Retrieving results..."
        python scripts/process_batch_evaluation.py \
            --batch_file "$BATCH_FILE" \
            --batch_id "$BATCH_ID" \
            --log_file "batch_prompts/extracted_answer/${FILE_NAME}_log.json"
        
        echo -e "${GREEN}✓ Answer extraction completed${NC}"
    fi
    
    # ============================================================
    # STEP 3: Correctness Check
    # ============================================================
    print_step "STEP 3: Correctness Evaluation"
    
    CORRECTNESS_OUTPUT="evaluation_outputs/extracted_answer/${FILE_NAME}_correctness.json"
    
    if [ -f "$CORRECTNESS_OUTPUT" ] && [ "$SKIP_COMPLETED" = true ]; then
        echo -e "${YELLOW}Skipping: Correctness check already completed${NC}"
    else
        echo "Generating batch prompts for correctness check..."
        python scripts/generate_batch_prompts_for_correctness_check.py \
            --file_name "$FILE_NAME" \
            --annotation_id "$ANNOTATION_ID" \
            --terminate_help
        
        BATCH_FILE="batch_prompts/extracted_answer/${FILE_NAME}_correctness.jsonl"
        
        echo "Submitting batch to OpenAI..."
        BATCH_OUTPUT=$(python scripts/process_batch_evaluation.py \
            --batch_file "$BATCH_FILE" \
            --model "$EVALUATOR_MODEL" 2>&1)
        
        BATCH_ID=$(echo "$BATCH_OUTPUT" | grep -oP 'batch_[a-zA-Z0-9]+' | head -1)
        
        if [ -z "$BATCH_ID" ]; then
            echo -e "${RED}Error: Failed to submit batch${NC}"
            exit 1
        fi
        
        echo -e "${GREEN}Batch submitted: $BATCH_ID${NC}"
        echo "Waiting for batch to complete..."
        
        # Poll for completion
        while true; do
            STATUS=$(python scripts/process_batch_evaluation.py \
                --batch_file "$BATCH_FILE" \
                --batch_id "$BATCH_ID" \
                --check_status 2>&1 | grep -oP '(?<=Status: )[a-z]+' || echo "unknown")
            
            if [ "$STATUS" = "completed" ]; then
                break
            elif [ "$STATUS" = "failed" ] || [ "$STATUS" = "expired" ]; then
                echo -e "${RED}Batch failed with status: $STATUS${NC}"
                exit 1
            fi
            
            echo "Status: $STATUS - waiting 30 seconds..."
            sleep 30
        done
        
        echo "Retrieving results..."
        python scripts/process_batch_evaluation.py \
            --batch_file "$BATCH_FILE" \
            --batch_id "$BATCH_ID" \
            --log_file "batch_prompts/extracted_answer/${FILE_NAME}_correctness_log.json"
        
        echo -e "${GREEN}✓ Correctness check completed${NC}"
    fi
    
    # ============================================================
    # STEP 4: Show Assistant Performance
    # ============================================================
    print_step "STEP 4: Assistant Performance Analysis"
    
    echo "Analyzing assistant model performance..."
    python scripts/show_assistant_performance.py \
        --annotation_id "$ANNOTATION_ID" \
        --file_name "$FILE_NAME" \
        --sort_by "correctness"
    
    # Also save results in different formats
    echo ""
    echo "Saving results in JSON format..."
    python scripts/show_assistant_performance.py \
        --annotation_id "$ANNOTATION_ID" \
        --file_name "$FILE_NAME" \
        --output_format "json" > "evaluation_outputs/assistant_performance_${FILE_NAME}.json"
    
    echo "Saving results in LaTeX format..."
    python scripts/show_assistant_performance.py \
        --annotation_id "$ANNOTATION_ID" \
        --file_name "$FILE_NAME" \
        --output_format "latex" > "evaluation_outputs/assistant_performance_${FILE_NAME}.tex"
    
    print_step "EVALUATION COMPLETE!"
    echo -e "${GREEN}All evaluation steps completed successfully${NC}"
    echo ""
    echo "Results saved to:"
    echo "  - Interaction ratings: $RATING_OUTPUT"
    echo "  - Extracted answers: $ANSWER_OUTPUT"
    echo "  - Correctness results: $CORRECTNESS_OUTPUT"
    echo "  - Performance summary: evaluation_outputs/assistant_performance_${FILE_NAME}.json"
}

# Run main function
main