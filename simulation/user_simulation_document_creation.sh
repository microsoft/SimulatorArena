#!/bin/bash

# ============================================================================
# User Simulation Batch Runner Script
# ============================================================================
# This script runs different configurations of user simulation experiments.
# Each section represents a different type of generation strategy.
# ============================================================================

# -----------------------------------------------------------------------------
# 1. BASIC GENERATION STRATEGIES (No User Profiles)
# -----------------------------------------------------------------------------

echo "========== BASIC ZERO-SHOT GENERATION =========="
echo "Simple generation without chain-of-thought reasoning"
echo ""
echo "Run $i: Zero-shot generation"
python user_simulation_document_creation.py --version=zero-shot
echo ""

echo "========== ZERO-SHOT WITH CHAIN-OF-THOUGHT =========="
echo "Generation with step-by-step reasoning"
echo ""
echo "Run $i: Zero-shot CoT generation"
python user_simulation_document_creation.py --version=zero-shot-cot
echo ""

# -----------------------------------------------------------------------------
# 2. USER PROFILE-BASED GENERATION (Different Profile Types)
# -----------------------------------------------------------------------------

echo "========== USER PROFILE: WRITING STYLE =========="
echo "Generation based on user's writing style characteristics"
echo ""
echo "Run $i: User profile with writing style"
python user_simulation_document_creation.py \
    --version=zero-shot-cot-user-profile \
    --user_profile_version=writing_style
echo ""

echo "========== USER PROFILE: INTERACTION STYLE =========="
echo "Generation based on user's interaction patterns"
echo ""
echo "Run: User profile with interaction style"
python user_simulation_document_creation.py \
    --version=zero-shot-cot-user-profile \
    --user_profile_version=interaction_style
echo ""

echo "========== USER PROFILE: PREFERENCE =========="
echo "Generation based on user's content preferences"
echo ""
for i in {1..4}; do
    echo "Run $i: User profile with preference"
    python user_simulation_document_creation.py \
        --version=zero-shot-cot-user-profile \
        --user_profile_version=preference
    echo ""
done

echo "========== USER PROFILE: COMBINED WRITING & INTERACTION & PREFERENCE =========="
echo "Generation combining writing style and interaction patterns and preference"
echo ""
echo "Run: User profile with combined writing and interaction style and preference"
python user_simulation_document_creation.py \
    --version=zero-shot-cot-user-profile \
    --user_profile_version=preference_and_writing_interaction_style
echo ""

# -----------------------------------------------------------------------------
# 3. LENGTH-CONTROLLED GENERATION
# -----------------------------------------------------------------------------

echo "========== BASIC LENGTH CONTROL =========="
echo "Generation with output length constraints"
echo ""
echo "Run: Zero-shot CoT with length control"
python user_simulation_document_creation.py \
    --version=zero-shot-cot-length-control \
    --length_control
echo ""

# -----------------------------------------------------------------------------
# 4. ADVANCED: USER PROFILES WITH LENGTH CONTROL
# -----------------------------------------------------------------------------

echo "========== ADVANCED: WRITING STYLE + LENGTH CONTROL =========="
echo "Combines user writing style with length constraints"
echo ""
for i in {1..3}; do
    echo "Run $i: Writing style profile with length control"
    python user_simulation_document_creation.py \
        --version=zero-shot-cot-user-profile-length-control \
        --length_control \
        --user_profile_version=writing_style
    echo ""
done

echo "========== ADVANCED: INTERACTION STYLE + LENGTH CONTROL =========="
echo "Combines user interaction patterns with length constraints"
echo ""
for i in {1..3}; do
    echo "Run $i: Interaction style profile with length control"
    python user_simulation_document_creation.py \
        --version=zero-shot-cot-user-profiles-length-control \
        --length_control \
        --user_profile_version=interaction_style
    echo ""
done

echo "========== ADVANCED: PREFERENCE + LENGTH CONTROL =========="
echo "User preferences with length constraints (currently active)"
echo ""
for i in {1..1}; do
    echo "Run $i: Preference profile with length control"
    python user_simulation_document_creation.py \
        --version=zero-shot-cot-user-profile-length-control \
        --length_control \
        --user_profile_version=preference
    echo ""
done

echo "========== ADVANCED: PREFERENCE + COMBINED STYLE + LENGTH CONTROL =========="
echo "Full combination of preferences, styles, and length constraints"
echo ""
for i in {1..3}; do
    echo "Run $i: Full profile combination with length control"
    python user_simulation_document_creation.py \
        --version=zero-shot-cot-user-profile-length-control \
        --length_control \
        --user_profile_version=preference_and_writing_interaction_style
    echo ""
done

# -----------------------------------------------------------------------------
# 5. EXPERIMENTAL FEATURES (Uncomment to use)
# -----------------------------------------------------------------------------

echo "========== REFINEMENT EXPERIMENTS =========="
echo "Generation with iterative refinement"
echo ""
python user_simulation_document_creation.py \
    --version=zero-shot-cot \
    --refinement \
    --refinement_version=v1

echo "========== CUSTOM USER MODEL =========="
echo "Using different language models for user simulation"
echo ""
python user_simulation_document_creation.py \
    --version=zero-shot-cot \
    --user_model=qwen3-8b

echo "============================================================================"
echo "All experiments completed!"
echo "============================================================================"
