#!/bin/bash

# ============================================================================
# Assistant Model Evaluation Script for Document Creation
# ============================================================================
# This script evaluates the performance of AI assistant models for document creation
# Used when testing new assistant models with benchmarking mode
# 
# IMPORTANT: Document extraction must happen BEFORE document rating
# ============================================================================

set -e  # Exit on error

# Default values
FILE_NAME=""
ANNOTATION_ID="document_creation_annotations"
EVALUATOR_MODEL="gpt-4o-2024-11-20"
SKIP_COMPLETED=true
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
            echo "  --annotation_id      Annotation dataset ID (default: document_creation_annotations)"
            echo "  --evaluator_model    Model for evaluation (default: gpt-4o-2024-11-20)"
            echo "  --no-skip           Don't skip already completed evaluations"
            echo "  --verbose           Show detailed output"
            echo "  --wait_interval     Polling interval in seconds (default: 30)"
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
    
    # Change to scripts directory for running Python scripts
    cd scripts
    
    # Check for terminated conversations
    TERMINATED_FILE="../../../simulation/terminated_conversations/${ANNOTATION_ID}/${FILE_NAME}.json"
    if ! check_file "$TERMINATED_FILE"; then
        echo -e "${YELLOW}Warning: Terminated conversations not found${NC}"
        echo "Please run termination detection first:"
        echo "  cd ../../../simulation"
        echo "  python terminate_conversation_document_creation.py --annotation_id $ANNOTATION_ID --simulation_path output/${ANNOTATION_ID}/${FILE_NAME}.json"
        exit 1
    fi
    
    # ============================================================
    # STEP 1: Document Extraction (MUST BE FIRST!)
    # ============================================================
    print_step "STEP 1: Document Extraction"
    
    DOCUMENT_OUTPUT="../evaluation_outputs/extracted_document/${FILE_NAME}.json"
    
    if [ -f "$DOCUMENT_OUTPUT" ] && [ "$SKIP_COMPLETED" = true ]; then
        echo -e "${YELLOW}Skipping: Document extraction already completed${NC}"
    else
        echo "Generating batch prompts for document extraction..."
        python generate_batch_prompts_for_document_extraction.py \
            --file_name "$FILE_NAME" \
            --annotation_id "$ANNOTATION_ID" \
            --terminate_help \
            --evaluator_model "$EVALUATOR_MODEL"
        
        BATCH_FILE="../batch_prompts/extracted_document/${FILE_NAME}.jsonl"
        
        if [ ! -f "$BATCH_FILE" ]; then
            echo -e "${YELLOW}No documents to extract (all already processed)${NC}"
        else
            echo "Submitting batch to OpenAI for document extraction..."
            python process_batch_evaluation.py \
                --batch_file "$BATCH_FILE" \
                --model "$EVALUATOR_MODEL" \
                --poll_interval $WAIT_INTERVAL \
                --wait
            
            echo -e "${GREEN}✓ Document extraction completed${NC}"
        fi
    fi
    
    # ============================================================
    # STEP 2: Document Rating (requires extracted documents)
    # ============================================================
    print_step "STEP 2: Document Quality Rating"
    
    DOCUMENT_RATING_OUTPUT="../evaluation_outputs/document_rating/${FILE_NAME}.json"
    
    if [ -f "$DOCUMENT_RATING_OUTPUT" ] && [ "$SKIP_COMPLETED" = true ]; then
        echo -e "${YELLOW}Skipping: Document rating already completed${NC}"
    else
        echo "Generating batch prompts for document rating..."
        python generate_batch_prompts_for_rating.py \
            --file_name "$FILE_NAME" \
            --annotation_id "$ANNOTATION_ID" \
            --aspect "document" \
            --evaluator_model "$EVALUATOR_MODEL"
        
        BATCH_FILE="../batch_prompts/document_rating/${FILE_NAME}.jsonl"
        
        if [ ! -f "$BATCH_FILE" ]; then
            echo -e "${YELLOW}No documents to rate${NC}"
        else
            echo "Submitting batch to OpenAI for document rating..."
            python process_batch_evaluation.py \
                --batch_file "$BATCH_FILE" \
                --model "$EVALUATOR_MODEL" \
                --poll_interval $WAIT_INTERVAL \
                --wait
            
            echo -e "${GREEN}✓ Document rating completed${NC}"
        fi
    fi
    
    # ============================================================
    # STEP 3: Interaction Rating
    # ============================================================
    print_step "STEP 3: Interaction Quality Rating"
    
    INTERACTION_RATING_OUTPUT="../evaluation_outputs/interaction_rating/${FILE_NAME}.json"
    
    if [ -f "$INTERACTION_RATING_OUTPUT" ] && [ "$SKIP_COMPLETED" = true ]; then
        echo -e "${YELLOW}Skipping: Interaction rating already completed${NC}"
    else
        echo "Generating batch prompts for interaction rating..."
        python generate_batch_prompts_for_rating.py \
            --file_name "$FILE_NAME" \
            --annotation_id "$ANNOTATION_ID" \
            --aspect "interaction" \
            --evaluator_model "$EVALUATOR_MODEL"
        
        BATCH_FILE="../batch_prompts/interaction_rating/${FILE_NAME}.jsonl"
        
        if [ ! -f "$BATCH_FILE" ]; then
            echo -e "${YELLOW}No interactions to rate${NC}"
        else
            echo "Submitting batch to OpenAI for interaction rating..."
            python process_batch_evaluation.py \
                --batch_file "$BATCH_FILE" \
                --model "$EVALUATOR_MODEL" \
                --poll_interval $WAIT_INTERVAL \
                --wait
            
            echo -e "${GREEN}✓ Interaction rating completed${NC}"
        fi
    fi
    
    # ============================================================
    # STEP 4: Show Assistant Performance
    # ============================================================
    print_step "STEP 4: Assistant Performance Analysis"
    
    echo "Analyzing assistant model performance..."
    echo ""
    echo "Performance by Document Quality:"
    python show_assistant_performance.py \
        --file_name "$FILE_NAME" \
        --annotation_id "$ANNOTATION_ID" \
        --sort_by "document"
    
    echo ""
    echo "Performance by Interaction Quality:"
    python show_assistant_performance.py \
        --file_name "$FILE_NAME" \
        --annotation_id "$ANNOTATION_ID" \
        --sort_by "interaction"
    
    echo ""
    echo "Performance by Combined Score:"
    python show_assistant_performance.py \
        --file_name "$FILE_NAME" \
        --annotation_id "$ANNOTATION_ID" \
        --sort_by "combined"
    
    # Export results
    echo ""
    echo "Exporting results..."
    python show_assistant_performance.py \
        --file_name "$FILE_NAME" \
        --annotation_id "$ANNOTATION_ID" \
        --sort_by "combined" \
        --export "../evaluation_outputs/assistant_performance_${FILE_NAME}.json"
    
    print_step "EVALUATION COMPLETE!"
    echo -e "${GREEN}All evaluation steps completed successfully${NC}"
    echo ""
    echo "Results saved to:"
    echo "  - Extracted documents: evaluation_outputs/extracted_document/${FILE_NAME}.json"
    echo "  - Document ratings: evaluation_outputs/document_rating/${FILE_NAME}.json"
    echo "  - Interaction ratings: evaluation_outputs/interaction_rating/${FILE_NAME}.json"
    echo "  - Performance summary: evaluation_outputs/assistant_performance_${FILE_NAME}.json"
}

# Run main function
main