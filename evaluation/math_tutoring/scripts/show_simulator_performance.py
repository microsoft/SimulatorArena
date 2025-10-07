#!/usr/bin/env python3
"""
Show simulator performance metrics comparing simulator vs human evaluation.
Based on:
- math_tutoring/user_simulation/extrinsic_evaluation_rating_analysis.ipynb (correlation)
- math_tutoring/user_simulation/extrinsic_evaluation_essence_analysis.ipynb (F1)

This script evaluates how well the simulator matches human judgment in two aspects:
1. Rating Correlation: How well simulator ratings correlate with human ratings
2. Essence (F1): How well simulator predicts correctness matching human correctness
"""

import json
import argparse
import sys
import statistics
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import scipy.stats as stats
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

def z_normalize(score: float, score_list: List[float]) -> float:
    """
    Returns the z-normalized value of 'score' based on the mean and standard 
    deviation of 'score_list'. If the standard deviation is 0 (all scores the same),
    returns 0.
    """
    if len(score_list) < 2:
        return 0
    
    mean_val = statistics.mean(score_list)
    stdev_val = statistics.stdev(score_list)

    if stdev_val == 0:
        return 0
    else:
        return (score - mean_val) / stdev_val

def load_evaluation_data(
    annotation_id: str,
    file_names: List[str],
    normalize: bool = True
) -> Tuple[Dict, Dict, List[Dict]]:
    """
    Load evaluation data for rating analysis from SimulatorArena evaluation outputs.
    
    Returns:
        Tuple of (rating_data_dict, correctness_data_dict, annotations)
    """
    # Load annotations from SimulatorArena data folder
    annotations_path = Path(__file__).parent.parent.parent.parent / "data" / "math_tutoring_annotations.json"
    if not annotations_path.exists():
        print(f"Error: {annotations_path} not found.")
        print("\nThe math tutoring annotations need to be generated first.")
        print("Please follow these steps:")
        print("1. Obtain the MATH dataset and place it in SimulatorArena/data/MATH/")
        print("2. Run: cd SimulatorArena/data && python load_math_data.py")
        print("\nSee SimulatorArena/data/README.md for detailed instructions.")
        sys.exit(1)
    
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    # Load rating data from SimulatorArena evaluation outputs
    rating_data_dict = {}
    for file_name in file_names:
        rating_path = Path(__file__).parent.parent / "evaluation_outputs" / "interaction_rating" / f"{file_name}.json"
        if not rating_path.exists():
            print(f"Warning: Cannot find rating file at {rating_path}")
            continue
        
        with open(rating_path, 'r') as f:
            data = json.load(f)
            # Handle format with "evaluations" wrapper
            if "evaluations" in data:
                data = data["evaluations"]
            rating_data_dict[file_name] = data
    
    # Load correctness data for essence analysis from SimulatorArena evaluation outputs
    correctness_data_dict = {}
    for file_name in file_names:
        correctness_path = Path(__file__).parent.parent / "evaluation_outputs" / "extracted_answer" / f"{file_name}_correctness.json"
        if not correctness_path.exists():
            print(f"Warning: Cannot find correctness file at {correctness_path}")
            continue

        with open(correctness_path, 'r') as f:
            data = json.load(f)
            # Handle format with "answers" wrapper
            if "answers" in data:
                data = data["answers"]
            correctness_data_dict[file_name] = data
    
    return rating_data_dict, correctness_data_dict, annotations

def calculate_rating_correlations(
    rating_data_dict: Dict,
    annotations: List[Dict],
    normalize: bool = True,
    desired_models: List[str] = None
) -> Dict[str, Dict]:
    """
    Calculate rating correlations at different levels.
    
    Returns:
        Dictionary with correlation metrics for each file
    """
    # Default desired models if not specified
    if desired_models is None:
        desired_models = ['phi-3-small', 'phi-3-medium', 'llama-3-1-8b', 'llama-3-1-70b', 
                         'mistral-large-latest', 'gpt-4o-mini', 'gpt-4-turbo', 
                         'claude-3-5-sonnet-20240620', 'gpt-4o']
    
    # Build user rating normalization data
    user_rating_dict = {}
    for annotation in annotations:
        worker_id = annotation['workerId']
        if worker_id not in user_rating_dict:
            user_rating_dict[worker_id] = {"overall_rating": []}
        user_rating_dict[worker_id]["overall_rating"].append(int(annotation["overall_rating"]))
    
    # Group ratings for normalization
    user_rating_group_less_3_dict = {"overall_rating": []}
    user_rating_group_all_dict = {"overall_rating": []}
    
    for worker_id, rating_dict in user_rating_dict.items():
        user_rating_group_all_dict["overall_rating"].extend(rating_dict["overall_rating"])
        if len(rating_dict["overall_rating"]) < 3:
            user_rating_group_less_3_dict["overall_rating"].extend(rating_dict["overall_rating"])
    
    results = {}
    
    for file_name, data in rating_data_dict.items():
        instance_model_ratings = []
        instance_human_ratings = []
        model_ratings_by_model = {}
        model_ratings_by_model_difficulty = {}
        
        for model, model_dict in data.items():
            if model not in model_ratings_by_model:
                model_ratings_by_model[model] = {
                    "model_ratings": [],
                    "human_ratings": []
                }
            
            for problem_id, problem_dict in model_dict.items():
                for turker_key, result in problem_dict.items():
                    # Find matching annotation
                    difficulty_level = None
                    for annotation in annotations:
                        annotation_key = f"{annotation['username']}_{annotation['workerId']}_{annotation['user_id']}"
                        if (annotation['problem_id'] == int(problem_id) and 
                            annotation["model"] == model and 
                            annotation_key == turker_key):
                            
                            human_rating = int(annotation["overall_rating"])
                            difficulty_level = annotation.get("level", "Unknown")
                            
                            if normalize:
                                worker_id = annotation['workerId']
                                if len(user_rating_dict[worker_id]["overall_rating"]) < 3:
                                    human_rating = z_normalize(human_rating, user_rating_group_less_3_dict["overall_rating"])
                                else:
                                    human_rating = z_normalize(human_rating, user_rating_dict[worker_id]["overall_rating"])
                            break
                    
                    try:
                        model_rating = int(result["extracted_rating"])
                        instance_model_ratings.append(model_rating)
                        instance_human_ratings.append(human_rating)
                        
                        model_ratings_by_model[model]["model_ratings"].append(model_rating)
                        model_ratings_by_model[model]["human_ratings"].append(human_rating)
                        
                        # Add to model-difficulty grouping
                        if difficulty_level:
                            key = (model, difficulty_level)
                            if key not in model_ratings_by_model_difficulty:
                                model_ratings_by_model_difficulty[key] = {
                                    "model_ratings": [],
                                    "human_ratings": []
                                }
                            model_ratings_by_model_difficulty[key]["model_ratings"].append(model_rating)
                            model_ratings_by_model_difficulty[key]["human_ratings"].append(human_rating)
                    except:
                        continue
        
        # Calculate instance-level correlations
        if len(instance_model_ratings) > 1:
            spearman_instance = stats.spearmanr(instance_model_ratings, instance_human_ratings).correlation
            pearson_instance = stats.pearsonr(instance_model_ratings, instance_human_ratings)[0]
            kendall_instance = stats.kendalltau(instance_model_ratings, instance_human_ratings).correlation
        else:
            spearman_instance = pearson_instance = kendall_instance = None
        
        # Calculate intermediate-level correlations (model, difficulty)
        intermediate_avg_model_ratings = []
        intermediate_avg_human_ratings = []
        
        for (model, difficulty), ratings_dict in model_ratings_by_model_difficulty.items():
            # Only include desired models for intermediate level
            if model in desired_models and len(ratings_dict["model_ratings"]) > 0:
                avg_model = np.mean(ratings_dict["model_ratings"])
                avg_human = np.mean(ratings_dict["human_ratings"])
                intermediate_avg_model_ratings.append(avg_model)
                intermediate_avg_human_ratings.append(avg_human)
        
        if len(intermediate_avg_model_ratings) > 1:
            spearman_intermediate = stats.spearmanr(intermediate_avg_model_ratings, intermediate_avg_human_ratings).correlation
            pearson_intermediate = stats.pearsonr(intermediate_avg_model_ratings, intermediate_avg_human_ratings)[0]
            kendall_intermediate = stats.kendalltau(intermediate_avg_model_ratings, intermediate_avg_human_ratings).correlation
        else:
            spearman_intermediate = pearson_intermediate = kendall_intermediate = None
        
        # Calculate system-level correlations
        avg_model_ratings = []
        avg_human_ratings = []
        
        for model, ratings_dict in model_ratings_by_model.items():
            # Only include desired models for system level
            if model in desired_models and len(ratings_dict["model_ratings"]) > 0:
                avg_model = np.mean(ratings_dict["model_ratings"])
                avg_human = np.mean(ratings_dict["human_ratings"])
                avg_model_ratings.append(avg_model)
                avg_human_ratings.append(avg_human)
        
        if len(avg_model_ratings) > 1:
            spearman_system = stats.spearmanr(avg_model_ratings, avg_human_ratings).correlation
            pearson_system = stats.pearsonr(avg_model_ratings, avg_human_ratings)[0]
            kendall_system = stats.kendalltau(avg_model_ratings, avg_human_ratings).correlation
        else:
            spearman_system = pearson_system = kendall_system = None
        
        results[file_name] = {
            "instance_level": {
                "spearman": spearman_instance,
                "pearson": pearson_instance,
                "kendall": kendall_instance,
                "n": len(instance_model_ratings)
            },
            "intermediate_level": {  # New: (model, difficulty) level
                "spearman": spearman_intermediate,
                "pearson": pearson_intermediate,
                "kendall": kendall_intermediate,
                "n": len(intermediate_avg_model_ratings)
            },
            "system_level": {
                "spearman": spearman_system,
                "pearson": pearson_system,
                "kendall": kendall_system,
                "n": len(avg_model_ratings)
            }
        }
    
    return results

def calculate_essence_f1(
    correctness_data_dict: Dict,
    annotations: List[Dict]
) -> Tuple[Dict[str, Dict], Dict[str, List]]:
    """
    Calculate F1 scores for correctness prediction (essence).
    
    Returns:
        Tuple of (metrics_dict, predictions_dict)
        - metrics_dict: Dictionary with F1 metrics for each file
        - predictions_dict: Dictionary with predictions and true labels for each file
    """
    results = {}
    predictions_dict = {}  # Store predictions for McNemar test
    
    for file_name, file_data in correctness_data_dict.items():
        preds = []
        true_labels = []
        
        for model_name, model_data in file_data.items():
            for problem_id, problem_data in model_data.items():
                for turker_key, results_dict in problem_data.items():
                    # Find matching annotation
                    for annotation in annotations:
                        annotation_key = f"{annotation['username']}_{annotation['workerId']}_{annotation['user_id']}"
                        if (annotation['problem_id'] == int(problem_id) and 
                            annotation["model"] == model_name and 
                            annotation_key == turker_key):
                            
                            problem_1_correctness = annotation['problem_1_correctness']
                            break
                    
                    # Get predicted correctness
                    # Try new field name first, fallback to old
                    pred = results_dict.get('correctness', results_dict.get('extracted_answer', 'incorrect'))
                    # Normalize to lowercase for comparison
                    pred = pred.lower() if isinstance(pred, str) else 'incorrect'
                    if problem_1_correctness == "unknown":
                        problem_1_correctness = "incorrect"

                    preds.append(pred)
                    true_labels.append(problem_1_correctness)
        
        # Store predictions for McNemar test
        predictions_dict[file_name] = {
            "preds": preds,
            "true": true_labels
        }
        
        # Calculate confusion matrix
        tp = fp = fn = tn = 0
        for p, t in zip(preds, true_labels):
            if p == "correct" and t == "correct":
                tp += 1
            elif p == "correct" and t == "incorrect":
                fp += 1
            elif p == "incorrect" and t == "correct":
                fn += 1
            else:
                tn += 1
        
        # Calculate F1 scores
        # F1 for "correct" class
        prec_correct = tp / (tp + fp) if (tp + fp) else 0
        rec_correct = tp / (tp + fn) if (tp + fn) else 0
        f1_correct = 2 * prec_correct * rec_correct / (prec_correct + rec_correct) if (prec_correct + rec_correct) else 0
        
        # F1 for "incorrect" class
        prec_incorrect = tn / (tn + fn) if (tn + fn) else 0
        rec_incorrect = tn / (tn + fp) if (tn + fp) else 0
        f1_incorrect = 2 * prec_incorrect * rec_incorrect / (prec_incorrect + rec_incorrect) if (prec_incorrect + rec_incorrect) else 0
        
        # Macro and Micro F1
        macro_f1 = (f1_correct + f1_incorrect) / 2
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0
        
        # Count distribution
        true_counter = Counter(true_labels)
        pred_counter = Counter(preds)
        
        results[file_name] = {
            "f1_correct": f1_correct,
            "f1_incorrect": f1_incorrect,
            "macro_f1": macro_f1,
            "accuracy": accuracy,
            "confusion_matrix": {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn
            },
            "true_distribution": dict(true_counter),
            "pred_distribution": dict(pred_counter),
            "n": len(preds)
        }
    
    return results, predictions_dict

def calculate_mcnemar_test(
    predictions_dict: Dict[str, Dict],
    baseline_name: str,
    comparison_name: str
) -> Tuple[float, float, str]:
    """
    Perform McNemar's test between two simulators.
    
    Returns:
        Tuple of (statistic, p_value, interpretation)
    """
    if baseline_name not in predictions_dict or comparison_name not in predictions_dict:
        return None, None, "Models not found for comparison"
    
    y_true = predictions_dict[baseline_name]["true"]
    pred_A = predictions_dict[baseline_name]["preds"]
    pred_B = predictions_dict[comparison_name]["preds"]
    
    # Contingency counts: b = A wrong, B right | c = A right, B wrong
    b = c = 0
    for t, a, bpred in zip(y_true, pred_A, pred_B):
        if a != t and bpred == t:
            b += 1
        if a == t and bpred != t:
            c += 1
    
    # Create contingency table
    table = [[0, b], [c, 0]]
    
    try:
        from statsmodels.stats.contingency_tables import mcnemar
        from math import sqrt
        result = mcnemar(table, exact=True)  # exact binomial version
        statistic = sqrt(result.statistic) if result.statistic else 0
        p_value = result.pvalue
        
        # Interpretation
        if p_value < 0.001:
            interpretation = "Very strong evidence of difference"
        elif p_value < 0.01:
            interpretation = "Strong evidence of difference"
        elif p_value < 0.05:
            interpretation = "Moderate evidence of difference"
        else:
            interpretation = "No significant difference"
        
        return statistic, p_value, interpretation
    except ImportError:
        return None, None, "statsmodels not available for McNemar test"
    except Exception as e:
        return None, None, f"Error computing McNemar test: {e}"

def display_results(
    rating_correlations: Dict,
    essence_f1_scores: Dict,
    predictions_dict: Dict = None,
    output_format: str = "table"
):
    """
    Display the combined results.
    """
    if output_format == "table":
        print("\n" + "="*120)
        print("SIMULATOR PERFORMANCE METRICS")
        print("="*120)
        
        # Rating Correlation Results
        print("\n### RATING CORRELATION ANALYSIS ###")
        print("-"*120)
        print(f"{'Simulator':<40} {'Instance Level':^24} {'Intermediate Level':^24} {'System Level':^24}")
        print(f"{'':40} {'Spear':>7} {'Pears':>7} {'Kend':>7} {'Spear':>7} {'Pears':>7} {'Kend':>7} {'Spear':>7} {'Pears':>7} {'Kend':>7}")
        print("-"*120)
        
        for file_name in sorted(rating_correlations.keys()):
            corr = rating_correlations[file_name]
            inst = corr["instance_level"]
            inter = corr.get("intermediate_level", {})  # Intermediate level (model, difficulty)
            sys = corr["system_level"]
            
            # Truncate long file names
            display_name = file_name[:37] + "..." if len(file_name) > 40 else file_name
            
            # Handle missing intermediate values
            inter_spear = inter.get('spearman', 0) if inter else 0
            inter_pears = inter.get('pearson', 0) if inter else 0
            inter_kend = inter.get('kendall', 0) if inter else 0
            
            print(f"{display_name:<40} "
                  f"{inst['spearman']:>7.3f} {inst['pearson']:>7.3f} {inst['kendall']:>7.3f} "
                  f"{inter_spear:>7.3f} {inter_pears:>7.3f} {inter_kend:>7.3f} "
                  f"{sys['spearman']:>7.3f} {sys['pearson']:>7.3f} {sys['kendall']:>7.3f}")
        
        # F1 Score Results
        print("\n### ESSENCE (CORRECTNESS) F1 ANALYSIS ###")
        print("-"*120)
        print(f"{'Simulator':<40} {'F1 Correct':>12} {'F1 Incorrect':>12} {'Macro F1':>10} {'Accuracy':>10}")
        print("-"*120)
        
        for file_name in sorted(essence_f1_scores.keys()):
            f1 = essence_f1_scores[file_name]
            
            # Truncate long file names
            display_name = file_name[:37] + "..." if len(file_name) > 40 else file_name
            
            print(f"{display_name:<40} "
                  f"{f1['f1_correct']:>12.3f} {f1['f1_incorrect']:>12.3f} "
                  f"{f1['macro_f1']:>10.3f} {f1['accuracy']:>10.3f}")
        
        print("-"*120)
        
        # McNemar Test (if we have predictions and at least 2 simulators)
        if predictions_dict and len(predictions_dict) >= 2:
            file_names = list(predictions_dict.keys())
            if len(file_names) >= 2:
                # Use first as baseline, second as comparison (user can modify as needed)
                baseline = file_names[0]
                comparison = file_names[1] if len(file_names) > 1 else file_names[0]
                
                statistic, p_value, interpretation = calculate_mcnemar_test(
                    predictions_dict, baseline, comparison
                )
                
                if statistic is not None:
                    print(f"\n### MCNEMAR TEST (ESSENCE) ###")
                    print(f"Baseline: {baseline}")
                    print(f"Comparison: {comparison}")
                    print(f"McNemar statistic: {statistic:.3f}")
                    print(f"P-value: {p_value:.4f}")
                    print(f"Interpretation: {interpretation}")
                    print("-"*120)
        
        # Summary statistics
        if len(rating_correlations) > 0:
            all_instance_spearman = [v["instance_level"]["spearman"] for v in rating_correlations.values() if v["instance_level"]["spearman"]]
            all_intermediate_spearman = [v["intermediate_level"]["spearman"] for v in rating_correlations.values() if v.get("intermediate_level", {}).get("spearman")]
            all_system_spearman = [v["system_level"]["spearman"] for v in rating_correlations.values() if v["system_level"]["spearman"]]
            all_macro_f1 = [v["macro_f1"] for v in essence_f1_scores.values()]
            
            print(f"\nSummary:")
            if all_instance_spearman:
                print(f"  Avg Instance Spearman:      {np.mean(all_instance_spearman):.3f}")
            if all_intermediate_spearman:
                print(f"  Avg Intermediate Spearman:  {np.mean(all_intermediate_spearman):.3f}")
            if all_system_spearman:
                print(f"  Avg System Spearman:        {np.mean(all_system_spearman):.3f}")
            if all_macro_f1:
                print(f"  Avg Macro F1:               {np.mean(all_macro_f1):.3f}")
        
    elif output_format == "json":
        results = {
            "rating_correlations": rating_correlations,
            "essence_f1_scores": essence_f1_scores
        }
        print(json.dumps(results, indent=2, default=str))
    
    elif output_format == "latex":
        print("\n% LaTeX table for simulator performance")
        print("\\begin{table}[h]")
        print("\\centering")
        print("\\begin{tabular}{lccccccccc}")
        print("\\toprule")
        print("& \\multicolumn{3}{c}{Instance} & \\multicolumn{3}{c}{Intermediate} & \\multicolumn{3}{c}{System} \\\\")
        print("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10}")
        print("Simulator & Spear & Pears & Kend & Spear & Pears & Kend & Spear & Pears & Kend \\\\")
        print("\\midrule")
        
        for file_name in sorted(rating_correlations.keys()):
            corr = rating_correlations.get(file_name, {})
            
            # Clean file name for LaTeX
            display_name = file_name.replace("_", "\\_")
            if len(display_name) > 30:
                display_name = display_name[:27] + "..."
            
            inst = corr.get("instance_level", {})
            inter = corr.get("intermediate_level", {})
            sys = corr.get("system_level", {})
            
            inst_spear = inst.get("spearman", 0) if inst else 0
            inst_pears = inst.get("pearson", 0) if inst else 0
            inst_kend = inst.get("kendall", 0) if inst else 0
            
            inter_spear = inter.get("spearman", 0) if inter else 0
            inter_pears = inter.get("pearson", 0) if inter else 0
            inter_kend = inter.get("kendall", 0) if inter else 0
            
            sys_spear = sys.get("spearman", 0) if sys else 0
            sys_pears = sys.get("pearson", 0) if sys else 0
            sys_kend = sys.get("kendall", 0) if sys else 0
            
            print(f"{display_name} & {inst_spear:.3f} & {inst_pears:.3f} & {inst_kend:.3f} & "
                  f"{inter_spear:.3f} & {inter_pears:.3f} & {inter_kend:.3f} & "
                  f"{sys_spear:.3f} & {sys_pears:.3f} & {sys_kend:.3f} \\\\")
        
        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\caption{Simulator Performance: Rating Correlations at Three Levels}")
        print("\\end{table}")
        
        # Separate table for F1 scores
        print("\n% LaTeX table for F1 scores")
        print("\\begin{table}[h]")
        print("\\centering")
        print("\\begin{tabular}{lcccc}")
        print("\\toprule")
        print("Simulator & F1-Correct & F1-Incorrect & Macro-F1 & Accuracy \\\\")
        print("\\midrule")
        
        for file_name in sorted(essence_f1_scores.keys()):
            f1 = essence_f1_scores.get(file_name, {})
            
            # Clean file name for LaTeX
            display_name = file_name.replace("_", "\\_")
            if len(display_name) > 40:
                display_name = display_name[:37] + "..."
            
            f1_correct = f1.get("f1_correct", 0)
            f1_incorrect = f1.get("f1_incorrect", 0)
            macro_f1 = f1.get("macro_f1", 0)
            accuracy = f1.get("accuracy", 0)
            
            print(f"{display_name} & {f1_correct:.3f} & {f1_incorrect:.3f} & {macro_f1:.3f} & {accuracy:.3f} \\\\")
        
        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\caption{Simulator Performance: Essence F1 Scores}")
        print("\\end{table}")

def main():
    parser = argparse.ArgumentParser(
        description="Show simulator performance metrics (rating correlation and essence F1)"
    )
    parser.add_argument(
        "--annotation_id",
        type=str,
        default="math_tutoring_annotations",
        help="Annotation ID for the evaluation data (default: math_tutoring_annotations)"
    )
    parser.add_argument(
        "--file_names",
        type=str,
        nargs="+",
        default=["zero-shot-cot", "zero-shot-cot-user-profile-up-interaction_style-upsource_combined"],
        help="List of simulator file names to evaluate"
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=True,
        help="Normalize human ratings for correlation"
    )
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["table", "latex", "json"],
        default="table",
        help="Output format for results"
    )
    parser.add_argument(
        "--desired_models",
        type=str,
        nargs="*",
        help="List of models to include in intermediate and system level correlations"
    )
    parser.add_argument(
        "--export",
        type=str,
        help="Export results to JSON file"
    )

    args = parser.parse_args()
    
    # Load data
    try:
        rating_data_dict, correctness_data_dict, annotations = load_evaluation_data(
            args.annotation_id,
            args.file_names,
            args.normalize
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Calculate rating correlations
    rating_correlations = calculate_rating_correlations(
        rating_data_dict,
        annotations,
        args.normalize,
        args.desired_models
    )
    
    # Calculate essence F1 scores
    essence_f1_scores, predictions_dict = calculate_essence_f1(
        correctness_data_dict,
        annotations
    )
    
    # Display results
    display_results(
        rating_correlations,
        essence_f1_scores,
        predictions_dict,
        args.output_format
    )

    # Export if requested
    if args.export:
        export_data = {
            "annotation_id": args.annotation_id,
            "file_names": args.file_names,
            "normalize": args.normalize,
            "rating_correlations": {},
            "essence_f1_scores": {}
        }

        # Export rating correlations
        for sim_name, corr_data in rating_correlations.items():
            export_data["rating_correlations"][sim_name] = {
                "conversation_level": round(corr_data["conversation_level"], 4) if corr_data["conversation_level"] is not None else None,
                "intermediate_level": round(corr_data["intermediate_level"], 4) if corr_data["intermediate_level"] is not None else None,
                "system_level": round(corr_data["system_level"], 4) if corr_data["system_level"] is not None else None
            }

        # Export essence F1 scores
        for sim_name, f1_data in essence_f1_scores.items():
            export_data["essence_f1_scores"][sim_name] = {
                "f1": round(f1_data["f1"], 4),
                "precision": round(f1_data["precision"], 4),
                "recall": round(f1_data["recall"], 4),
                "support": f1_data["support"]
            }

        # Ensure parent directory exists
        export_path = Path(args.export)
        export_path.parent.mkdir(parents=True, exist_ok=True)

        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"\nResults exported to: {export_path}")

if __name__ == "__main__":
    main()