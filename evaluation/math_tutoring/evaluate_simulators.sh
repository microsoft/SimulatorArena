#!/bin/bash

# ============================================================================
# User Simulator Evaluation Script for Math Tutoring
# ============================================================================
# This script evaluates how well user simulators mimic human behavior
# Used when testing and developing new user simulators
# ============================================================================

set -e  # Exit on error

# Default values
FILE_NAME=""
ANNOTATION_ID="math_tutoring_annotations"
EVALUATOR_MODEL="gpt-5-mini"
SKIP_COMPLETED=true
COMPARE_WITH=""
VERBOSE=false
WAIT_INTERVAL=30

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
        --compare_with)
            COMPARE_WITH="$2"
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
        --wait_interval)
            WAIT_INTERVAL="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 --file_name <name> [options]"
            echo ""
            echo "Options:"
            echo "  --file_name          Name of simulation output file (required)"
            echo "  --annotation_id      Annotation dataset ID (default: math_tutoring_annotations)"
            echo "  --evaluator_model    Model for evaluation (default: gpt-5-mini)"
            echo "  --compare_with       Additional simulators to compare (comma-separated)"
            echo "  --no-skip           Don't skip already completed evaluations"
            echo "  --verbose           Show detailed output"
            echo "  --wait_interval     Polling interval in seconds (default: 30)"
            echo "  --help              Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --file_name zero-shot-cot"
            echo "  $0 --file_name zero-shot-cot --compare_with 'zero-shot,zero-shot-cot-user-profile_up-interaction_style'"
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
    print_step "EVALUATING USER SIMULATOR: $FILE_NAME"
    
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

    RATING_OUTPUT="evaluation_outputs/interaction_rating/${FILE_NAME}.json"

    if [ -f "$RATING_OUTPUT" ] && [ "$SKIP_COMPLETED" = true ]; then
        echo -e "${YELLOW}Skipping: Interaction rating already completed${NC}"
    else
        echo "Generating batch prompts for interaction rating..."
        python scripts/generate_batch_prompts_for_interaction_rating.py \
            --file_name "$FILE_NAME" \
            --annotation_id "$ANNOTATION_ID"

        BATCH_FILE="batch_prompts/interaction_rating/${FILE_NAME}.jsonl"

        echo "Submitting batch to OpenAI..."
        python scripts/process_batch_evaluation.py \
            --batch_file "$BATCH_FILE" \
            --model "$EVALUATOR_MODEL" \
            --poll_interval $WAIT_INTERVAL

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
            --annotation_id "$ANNOTATION_ID"
        
        BATCH_FILE="batch_prompts/extracted_answer/${FILE_NAME}.jsonl"

        echo "Submitting batch to OpenAI..."
        python scripts/process_batch_evaluation.py \
            --batch_file "$BATCH_FILE" \
            --model "$EVALUATOR_MODEL" \
            --poll_interval $WAIT_INTERVAL

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
            --annotation_id "$ANNOTATION_ID"
        
        BATCH_FILE="batch_prompts/extracted_answer/${FILE_NAME}_correctness.jsonl"

        echo "Submitting batch to OpenAI..."
        python scripts/process_batch_evaluation.py \
            --batch_file "$BATCH_FILE" \
            --model "$EVALUATOR_MODEL" \
            --poll_interval $WAIT_INTERVAL

        echo -e "${GREEN}✓ Correctness check completed${NC}"
    fi
    
    # ============================================================
    # STEP 4: Show Simulator Performance
    # ============================================================
    print_step "STEP 4: Simulator Performance Analysis"
    
    # Build file names list for comparison
    if [ -n "$COMPARE_WITH" ]; then
        # Convert comma-separated list to space-separated for Python
        FILE_NAMES="$FILE_NAME"
        IFS=',' read -ra COMPARE_ARRAY <<< "$COMPARE_WITH"
        for simulator in "${COMPARE_ARRAY[@]}"; do
            FILE_NAMES="$FILE_NAMES $(echo $simulator | xargs)"  # xargs to trim whitespace
        done
        
        echo "Comparing simulators: $FILE_NAMES"
        echo ""

        python scripts/show_simulator_performance.py \
            --annotation_id "$ANNOTATION_ID" \
            --file_names $FILE_NAMES \
            --evaluator_model "_${EVALUATOR_MODEL//-/_}"
    else
        echo "Analyzing simulator: $FILE_NAME"
        echo ""

        python scripts/show_simulator_performance.py \
            --annotation_id "$ANNOTATION_ID" \
            --file_names "$FILE_NAME" \
            --evaluator_model "_${EVALUATOR_MODEL//-/_}"
    fi
    
    # Save results in different formats
    echo ""
    echo "Saving detailed results..."
    
    # JSON format
    if [ -n "$COMPARE_WITH" ]; then
        OUTPUT_NAME="simulator_comparison_${FILE_NAME}"
    else
        OUTPUT_NAME="simulator_performance_${FILE_NAME}"
    fi
    
    # Ensure output directory exists
    OUTPUT_JSON="evaluation_outputs/${OUTPUT_NAME}.json"
    OUTPUT_TEX="evaluation_outputs/${OUTPUT_NAME}.tex"
    mkdir -p $(dirname "$OUTPUT_JSON")

    python scripts/show_simulator_performance.py \
        --annotation_id "$ANNOTATION_ID" \
        --file_names $FILE_NAMES \
        --evaluator_model "_${EVALUATOR_MODEL//-/_}" \
        --output_format "json" > "$OUTPUT_JSON"

    echo "Results saved to: $OUTPUT_JSON"

    # LaTeX format for papers
    python scripts/show_simulator_performance.py \
        --annotation_id "$ANNOTATION_ID" \
        --file_names $FILE_NAMES \
        --evaluator_model "_${EVALUATOR_MODEL//-/_}" \
        --output_format "latex" > "$OUTPUT_TEX"

    echo "LaTeX table saved to: $OUTPUT_TEX"
    
    print_step "EVALUATION COMPLETE!"
    echo -e "${GREEN}All evaluation steps completed successfully${NC}"
    echo ""
    echo "Key Metrics:"
    echo "  - Correlation with human ratings (instance/intermediate/system levels)"
    echo "  - F1 scores for correctness prediction"
    echo "  - Overall simulator fidelity"
    echo ""
    echo "Results saved to:"
    echo "  - Interaction ratings: $RATING_OUTPUT"
    echo "  - Extracted answers: $ANSWER_OUTPUT"
    echo "  - Correctness results: $CORRECTNESS_OUTPUT"
    echo "  - Performance summary: $OUTPUT_JSON"
    
    # Show baseline comparison hint
    if [ -z "$COMPARE_WITH" ]; then
        echo ""
        echo -e "${YELLOW}Tip: To compare with baseline simulators, use:${NC}"
        echo "  $0 --file_name $FILE_NAME --compare_with 'zero-shot,zero-shot-cot'"
    fi
}

# Run main function
main