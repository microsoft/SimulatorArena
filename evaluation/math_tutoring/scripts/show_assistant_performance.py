#!/usr/bin/env python3
"""
Show assistant (tutor model) performance metrics.
Based on math_tutoring/user_simulation/benchmark_results.ipynb

This script evaluates the performance of different assistant models in the math tutoring task.
Metrics include:
1. Average conversation turns
2. Average conversation rating (1-10 scale)
3. Answer correctness rate (percentage)
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

def load_data(
    annotation_id: str,
    file_name: str
) -> Tuple[Dict, Dict, Dict]:
    """
    Load the three required data files for evaluation from SimulatorArena outputs.
    
    Returns:
        Tuple of (answer_correctness_data, conversation_rating_data, terminated_conversations)
    """
    
    # Load answer correctness data from SimulatorArena evaluation outputs
    correctness_path = Path(__file__).parent.parent / "evaluation_outputs" / "extracted_answer" / f"{file_name}.json"
    if not correctness_path.exists():
        print(f"Error: Cannot find answer correctness file at {correctness_path}")
        sys.exit(1)
    
    with open(correctness_path, 'r') as f:
        answer_correctness_data = json.load(f)
        # Handle new format with "answers" wrapper
        if "answers" in answer_correctness_data:
            answer_correctness_data = answer_correctness_data["answers"]
    
    # Load conversation rating data from SimulatorArena evaluation outputs
    rating_path = Path(__file__).parent.parent / "evaluation_outputs" / "interaction_rating" / f"{file_name}.json"
    if not rating_path.exists():
        print(f"Error: Cannot find rating file at {rating_path}")
        sys.exit(1)
    
    with open(rating_path, 'r') as f:
        conversation_rating_data = json.load(f)
        # Handle new format with "evaluations" wrapper
        if "evaluations" in conversation_rating_data:
            conversation_rating_data = conversation_rating_data["evaluations"]
    
    # Load terminated conversations data from SimulatorArena simulation outputs
    terminated_path = Path(__file__).parent.parent.parent.parent / "simulation" / "terminated_conversations" / annotation_id / f"{file_name}.json"
    if not terminated_path.exists():
        print(f"Error: Cannot find terminated conversations file at {terminated_path}")
        print("Note: Terminated conversations data should be generated during simulation.")
        sys.exit(1)
    
    with open(terminated_path, 'r') as f:
        terminated_conversations = json.load(f)
    
    return answer_correctness_data, conversation_rating_data, terminated_conversations

def calculate_model_statistics(
    answer_correctness_data: Dict,
    conversation_rating_data: Dict,
    terminated_conversations: Dict
) -> List[Tuple[str, float, float, float, int]]:
    """
    Calculate performance statistics for each model.
    
    Returns:
        List of tuples (model_name, avg_turns, avg_rating, correctness_percentage, sample_count)
    """
    model_stats = []
    
    for model in terminated_conversations:
        conversations_turns = []
        answer_correctness = []
        conversation_ratings = []
        
        for problem_id in terminated_conversations[model]:
            for user_key in terminated_conversations[model][problem_id]:
                # Get conversation turns
                turns = terminated_conversations[model][problem_id][user_key].get("ending_turn_number")
                if turns is not None:
                    conversations_turns.append(turns)
                
                # Get conversation rating
                if (model in conversation_rating_data and 
                    problem_id in conversation_rating_data[model] and 
                    user_key in conversation_rating_data[model][problem_id]):
                    
                    rating_str = conversation_rating_data[model][problem_id][user_key].get("extracted_rating")
                    if rating_str and rating_str != "Error":
                        try:
                            rating = float(rating_str)
                            conversation_ratings.append(rating)
                        except ValueError:
                            pass
                
                # Get answer correctness
                if (model in answer_correctness_data and 
                    problem_id in answer_correctness_data[model] and 
                    user_key in answer_correctness_data[model][problem_id]):
                    
                    correctness = answer_correctness_data[model][problem_id][user_key].get("correctness", "incorrect")
                    answer_correctness.append(correctness)
        
        # Calculate averages
        if conversations_turns and conversation_ratings and answer_correctness:
            avg_turns = np.mean(conversations_turns)
            avg_rating = np.mean(conversation_ratings)
            correctness_rate = (sum(1 for x in answer_correctness if x == "correct") / len(answer_correctness)) * 100
            sample_count = len(conversation_ratings)
            
            model_stats.append((model, avg_turns, avg_rating, correctness_rate, sample_count))
    
    return model_stats

def display_results(
    model_stats: List[Tuple[str, float, float, float, int]],
    sort_by: str = "rating",
    output_format: str = "table"
):
    """
    Display the results in the specified format.
    
    Args:
        model_stats: List of model statistics
        sort_by: Metric to sort by ('rating', 'correctness', 'turns', 'model')
        output_format: Output format ('table', 'latex', 'json')
    """
    # Sort the results
    if sort_by == "rating":
        model_stats.sort(key=lambda x: x[2], reverse=True)
    elif sort_by == "correctness":
        model_stats.sort(key=lambda x: x[3], reverse=True)
    elif sort_by == "turns":
        model_stats.sort(key=lambda x: x[1])
    elif sort_by == "model":
        model_stats.sort(key=lambda x: x[0])
    
    if output_format == "table":
        print("\n" + "="*80)
        print("ASSISTANT MODEL PERFORMANCE METRICS")
        print("="*80)
        print(f"{'Model':<30} {'Avg Turns':>10} {'Avg Rating':>12} {'Correctness':>12} {'N':>5}")
        print("-"*80)
        
        for model, avg_turns, avg_rating, correctness, n in model_stats:
            print(f"{model:<30} {avg_turns:>10.1f} {avg_rating:>12.2f} {correctness:>11.1f}% {n:>5}")
        
        print("-"*80)
        
        # Calculate overall statistics
        all_ratings = [x[2] for x in model_stats]
        all_correctness = [x[3] for x in model_stats]
        all_turns = [x[1] for x in model_stats]
        
        print(f"\nSummary Statistics:")
        print(f"  Rating:      Mean={np.mean(all_ratings):.2f}, Std={np.std(all_ratings):.2f}, Range=[{min(all_ratings):.2f}, {max(all_ratings):.2f}]")
        print(f"  Correctness: Mean={np.mean(all_correctness):.1f}%, Std={np.std(all_correctness):.1f}%, Range=[{min(all_correctness):.1f}%, {max(all_correctness):.1f}%]")
        print(f"  Turns:       Mean={np.mean(all_turns):.1f}, Std={np.std(all_turns):.1f}, Range=[{min(all_turns):.1f}, {max(all_turns):.1f}]")
        
    elif output_format == "latex":
        print("\n% LaTeX table for assistant performance")
        print("\\begin{table}[h]")
        print("\\centering")
        print("\\begin{tabular}{lccc}")
        print("\\toprule")
        print("Model & Avg Rating & Correctness (\\%) & Avg Turns \\\\")
        print("\\midrule")
        
        for model, avg_turns, avg_rating, correctness, _ in model_stats:
            # Clean model name for LaTeX
            model_latex = model.replace("_", "\\_")
            print(f"{model_latex} & {avg_rating:.2f} & {correctness:.1f} & {avg_turns:.1f} \\\\")
        
        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\caption{Assistant Model Performance Metrics}")
        print("\\end{table}")
        
    elif output_format == "json":
        results = {
            "metrics": [
                {
                    "model": model,
                    "avg_turns": round(avg_turns, 2),
                    "avg_rating": round(avg_rating, 2),
                    "correctness_percentage": round(correctness, 2),
                    "sample_count": n
                }
                for model, avg_turns, avg_rating, correctness, n in model_stats
            ],
            "summary": {
                "total_models": len(model_stats),
                "avg_rating_across_models": round(np.mean([x[2] for x in model_stats]), 2),
                "avg_correctness_across_models": round(np.mean([x[3] for x in model_stats]), 2),
                "avg_turns_across_models": round(np.mean([x[1] for x in model_stats]), 2)
            }
        }
        print(json.dumps(results, indent=2))

def main():
    parser = argparse.ArgumentParser(
        description="Show assistant model performance metrics for math tutoring task"
    )
    parser.add_argument(
        "--annotation_id",
        type=str,
        default="good_annotations_50_benchmarking",
        help="Annotation ID for the evaluation data"
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default="zero-shot-cot-user-profile-up-interaction_style",
        help="Base file name for the evaluation data"
    )
    parser.add_argument(
        "--sort_by",
        type=str,
        choices=["rating", "correctness", "turns", "model"],
        default="rating",
        help="Metric to sort results by"
    )
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["table", "latex", "json"],
        default="table",
        help="Output format for results"
    )
    parser.add_argument(
        "--top_n",
        type=int,
        help="Show only top N models"
    )
    
    args = parser.parse_args()
    
    # Load data
    try:
        answer_correctness_data, conversation_rating_data, terminated_conversations = load_data(
            args.annotation_id,
            args.file_name
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Calculate statistics
    model_stats = calculate_model_statistics(
        answer_correctness_data,
        conversation_rating_data,
        terminated_conversations
    )
    
    if not model_stats:
        print("No model statistics could be calculated. Check your data files.")
        sys.exit(1)
    
    # Filter top N if specified
    if args.top_n:
        # Sort first to get top N
        if args.sort_by == "rating":
            model_stats.sort(key=lambda x: x[2], reverse=True)
        elif args.sort_by == "correctness":
            model_stats.sort(key=lambda x: x[3], reverse=True)
        elif args.sort_by == "turns":
            model_stats.sort(key=lambda x: x[1])
        
        model_stats = model_stats[:args.top_n]
    
    # Display results
    display_results(model_stats, args.sort_by, args.output_format)

if __name__ == "__main__":
    main()