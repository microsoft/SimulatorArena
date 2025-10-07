#!/usr/bin/env python3
"""
Show simulator performance metrics for document creation task.
This script evaluates correlation between simulator and human ratings at multiple levels.

Unlike math tutoring, document creation uses correlation metrics for both aspects (no F1 scores).
"""

import json
import argparse
import statistics
import scipy.stats as stats
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tabulate import tabulate
import numpy as np

def load_human_annotations(annotation_id: str) -> List[Dict]:
    """Load human annotations (ground truth)."""
    annotation_path = Path(__file__).parent.parent.parent.parent / "simulation" / "data" / f"{annotation_id}.json"
    
    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
    
    with open(annotation_path, 'r') as f:
        return json.load(f)

def load_simulator_evaluations(
    file_name: str,
    aspect: str  # "document" or "interaction"
) -> Optional[Dict]:
    """Load simulator evaluation results."""
    eval_dir = Path(__file__).parent.parent / "evaluation_outputs"
    
    if aspect == "document":
        eval_file = eval_dir / "document_rating" / f"{file_name}.json"
    else:
        eval_file = eval_dir / "interaction_rating" / f"{file_name}.json"
    
    if not eval_file.exists():
        return None
    
    with open(eval_file, 'r') as f:
        data = json.load(f)
        # Handle both direct format and wrapped format
        if "evaluations" in data:
            return data["evaluations"]
        return data

def z_normalize(score: float, score_list: List[float]) -> float:
    """
    Returns the z-normalized value of 'score' based on the mean and 
    standard deviation of 'score_list'. If the standard deviation is 0,
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

def prepare_rating_data(
    simulator_data: Dict,
    annotations: List[Dict],
    aspect: str,
    normalize: bool = True
) -> Tuple[Dict, Dict, Dict]:
    """
    Prepare rating data for correlation analysis.
    
    Returns:
        Tuple of (instance_level_data, intermediate_level_data, system_level_data)
    """
    # Document type mapping
    document_type_dict = {
        "blog post": "Blog Post",
        "email": "Email/Letter",
        "creative writing": "Creative Writing",
    }
    
    # Build user rating dictionary for normalization
    user_rating_dict = {}
    for annotation in annotations:
        worker_id = annotation['workerId']
        if worker_id not in user_rating_dict:
            user_rating_dict[worker_id] = {
                "overall_document_rating": [],
                "overall_interaction_rating": [],
            }
        user_rating_dict[worker_id]["overall_document_rating"].append(
            int(annotation["overall_document_rating"])
        )
        user_rating_dict[worker_id]["overall_interaction_rating"].append(
            int(annotation["overall_interaction_rating"])
        )
    
    # Create groups for normalization
    user_rating_group_less_3_dict = {
        "overall_document_rating": [],
        "overall_interaction_rating": [],
    }
    
    for worker_id, rating_dict in user_rating_dict.items():
        if len(rating_dict["overall_document_rating"]) < 3:
            user_rating_group_less_3_dict["overall_document_rating"].extend(
                rating_dict["overall_document_rating"]
            )
            user_rating_group_less_3_dict["overall_interaction_rating"].extend(
                rating_dict["overall_interaction_rating"]
            )
    
    # Prepare data at different levels
    instance_data = {
        "model_ratings": [],
        "human_ratings": []
    }
    
    intermediate_data = {}  # Key: (model, document_type)
    system_data = {}  # Key: model
    
    # Process simulator data
    for model, model_dict in simulator_data.items():
        if model not in system_data:
            system_data[model] = {
                "model_ratings": [],
                "human_ratings": []
            }
        
        for document_type, document_dict in model_dict.items():
            intermediate_key = (model, document_type)
            if intermediate_key not in intermediate_data:
                intermediate_data[intermediate_key] = {
                    "model_ratings": [],
                    "human_ratings": []
                }
            
            for intent, intent_dict in document_dict.items():
                for worker_id, results in intent_dict.items():
                    # Find matching human annotation
                    human_rating = None
                    for annotation in annotations:
                        if (annotation["workerId"] == worker_id and 
                            document_type == document_type_dict.get(annotation["document_type"]) and
                            intent == annotation["intent"] and 
                            model == annotation["model"]):
                            
                            human_rating = int(annotation[f"overall_{aspect}_rating"])
                            
                            # Normalize if requested
                            if normalize:
                                if len(user_rating_dict[worker_id][f"overall_{aspect}_rating"]) < 3:
                                    human_rating = z_normalize(
                                        human_rating,
                                        user_rating_group_less_3_dict[f"overall_{aspect}_rating"]
                                    )
                                else:
                                    human_rating = z_normalize(
                                        human_rating,
                                        user_rating_dict[worker_id][f"overall_{aspect}_rating"]
                                    )
                            break
                    
                    if human_rating is None:
                        continue
                    
                    # Extract model rating
                    try:
                        if isinstance(results, dict):
                            if "extracted_rating" in results:
                                model_rating = float(results["extracted_rating"])
                            elif "rating" in results:
                                model_rating = float(results["rating"])
                            else:
                                continue
                        else:
                            model_rating = float(results)
                        
                        model_rating = min(10, model_rating)
                        
                        # Add to all levels
                        instance_data["model_ratings"].append(model_rating)
                        instance_data["human_ratings"].append(human_rating)
                        
                        intermediate_data[intermediate_key]["model_ratings"].append(model_rating)
                        intermediate_data[intermediate_key]["human_ratings"].append(human_rating)
                        
                        system_data[model]["model_ratings"].append(model_rating)
                        system_data[model]["human_ratings"].append(human_rating)
                        
                    except (ValueError, TypeError, KeyError):
                        continue
    
    return instance_data, intermediate_data, system_data

def calculate_correlations(model_ratings: List[float], human_ratings: List[float]) -> Dict:
    """Calculate Spearman, Pearson, and Kendall correlations."""
    if len(model_ratings) < 2 or len(human_ratings) < 2:
        return {
            "spearman": None,
            "pearson": None,
            "kendall": None,
            "n": len(model_ratings)
        }
    
    try:
        spearman_corr = stats.spearmanr(model_ratings, human_ratings).correlation
        pearson_corr = stats.pearsonr(model_ratings, human_ratings)[0]
        kendall_corr = stats.kendalltau(model_ratings, human_ratings).correlation
        
        return {
            "spearman": spearman_corr,
            "pearson": pearson_corr,
            "kendall": kendall_corr,
            "n": len(model_ratings)
        }
    except Exception as e:
        print(f"Warning: Could not calculate correlations: {e}")
        return {
            "spearman": None,
            "pearson": None,
            "kendall": None,
            "n": len(model_ratings)
        }

def display_correlation_results(
    instance_corr: Dict,
    intermediate_corr: Dict,
    system_corr: Dict,
    aspect: str
):
    """Display correlation results in a formatted table."""
    print("\n" + "="*80)
    print(f"SIMULATOR PERFORMANCE - {aspect.upper()} ASPECT")
    print("="*80)
    
    # Instance-level results
    print("\n1. INSTANCE-LEVEL CORRELATION")
    print("-"*40)
    print("(Individual conversation correlations)")
    
    if instance_corr["spearman"] is not None:
        headers = ["Metric", "Value", "N"]
        table_data = [
            ["Spearman ρ", f"{instance_corr['spearman']:.3f}", instance_corr['n']],
            ["Pearson r", f"{instance_corr['pearson']:.3f}", ""],
            ["Kendall τ", f"{instance_corr['kendall']:.3f}", ""]
        ]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    else:
        print("Insufficient data for correlation calculation")
    
    # Intermediate-level results
    print("\n2. INTERMEDIATE-LEVEL CORRELATION")
    print("-"*40)
    print("(Grouped by model and document type)")
    
    if intermediate_corr["spearman"] is not None:
        headers = ["Metric", "Value", "N Groups"]
        table_data = [
            ["Spearman ρ", f"{intermediate_corr['spearman']:.3f}", intermediate_corr['n']],
            ["Pearson r", f"{intermediate_corr['pearson']:.3f}", ""],
            ["Kendall τ", f"{intermediate_corr['kendall']:.3f}", ""]
        ]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    else:
        print("Insufficient data for correlation calculation")
    
    # System-level results
    print("\n3. SYSTEM-LEVEL CORRELATION")
    print("-"*40)
    print("(Grouped by model only)")
    
    if system_corr["spearman"] is not None:
        headers = ["Metric", "Value", "N Models"]
        table_data = [
            ["Spearman ρ", f"{system_corr['spearman']:.3f}", system_corr['n']],
            ["Pearson r", f"{system_corr['pearson']:.3f}", ""],
            ["Kendall τ", f"{system_corr['kendall']:.3f}", ""]
        ]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # Provide interpretation
        print("\nInterpretation Guide:")
        print("  Spearman ρ: Rank correlation (robust to outliers)")
        print("  Pearson r:  Linear correlation")
        print("  Kendall τ:  Rank correlation (handles ties well)")
        print("\nCorrelation Strength:")
        print("  0.70-1.00: Strong correlation")
        print("  0.40-0.69: Moderate correlation")
        print("  0.20-0.39: Weak correlation")
        print("  0.00-0.19: Very weak/no correlation")
    else:
        print("Insufficient data for correlation calculation")
    
    print("="*80)

def main():
    parser = argparse.ArgumentParser(
        description="Display simulator performance metrics for document creation task."
    )
    
    parser.add_argument(
        "--file_name",
        type=str,
        required=True,
        help="Name of the evaluation file (without extension)"
    )
    parser.add_argument(
        "--annotation_id",
        type=str,
        default="document_creation_annotations",
        help="Annotation dataset ID for ground truth"
    )
    parser.add_argument(
        "--aspect",
        type=str,
        choices=["document", "interaction", "both"],
        default="both",
        help="Which aspect to evaluate (default: both)"
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=True,
        help="Normalize human ratings per annotator (default: True)"
    )
    parser.add_argument(
        "--no_normalize",
        action="store_false",
        dest="normalize",
        help="Do not normalize human ratings"
    )
    parser.add_argument(
        "--export",
        type=str,
        help="Export results to JSON file"
    )
    
    args = parser.parse_args()
    
    # Load human annotations
    print(f"Loading human annotations from: {args.annotation_id}")
    try:
        annotations = load_human_annotations(args.annotation_id)
        print(f"Loaded {len(annotations)} annotations")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return
    
    export_data = {
        "file_name": args.file_name,
        "annotation_id": args.annotation_id,
        "normalize": args.normalize,
        "results": {}
    }
    
    # Process each aspect
    aspects_to_process = ["document", "interaction"] if args.aspect == "both" else [args.aspect]
    
    for aspect in aspects_to_process:
        print(f"\nProcessing {aspect} aspect...")
        
        # Load simulator evaluations
        simulator_data = load_simulator_evaluations(args.file_name, aspect)
        
        if not simulator_data:
            print(f"WARNING: No {aspect} evaluation data found for {args.file_name}")
            continue
        
        # Prepare data
        instance_data, intermediate_data, system_data = prepare_rating_data(
            simulator_data,
            annotations,
            aspect,
            normalize=args.normalize
        )
        
        # Calculate instance-level correlations
        instance_corr = calculate_correlations(
            instance_data["model_ratings"],
            instance_data["human_ratings"]
        )
        
        # Calculate intermediate-level correlations (group by model, document_type)
        intermediate_model_ratings = []
        intermediate_human_ratings = []
        
        for (model, doc_type), data in intermediate_data.items():
            if len(data["model_ratings"]) > 0 and len(data["human_ratings"]) > 0:
                intermediate_model_ratings.append(np.mean(data["model_ratings"]))
                intermediate_human_ratings.append(np.mean(data["human_ratings"]))
        
        intermediate_corr = calculate_correlations(
            intermediate_model_ratings,
            intermediate_human_ratings
        )
        
        # Calculate system-level correlations (group by model)
        system_model_ratings = []
        system_human_ratings = []
        
        for model, data in system_data.items():
            if len(data["model_ratings"]) > 0 and len(data["human_ratings"]) > 0:
                system_model_ratings.append(np.mean(data["model_ratings"]))
                system_human_ratings.append(np.mean(data["human_ratings"]))
        
        system_corr = calculate_correlations(
            system_model_ratings,
            system_human_ratings
        )
        
        # Display results
        display_correlation_results(
            instance_corr,
            intermediate_corr,
            system_corr,
            aspect
        )
        
        # Store for export
        export_data["results"][aspect] = {
            "instance_level": instance_corr,
            "intermediate_level": intermediate_corr,
            "system_level": system_corr
        }
    
    # Export if requested
    if args.export:
        # Ensure parent directory exists
        export_path = Path(args.export)
        export_path.parent.mkdir(parents=True, exist_ok=True)

        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"\nResults exported to: {export_path}")
    
    # Summary comparison if both aspects were evaluated
    if len(aspects_to_process) == 2:
        print("\n" + "="*80)
        print("COMPARATIVE SUMMARY")
        print("="*80)
        
        headers = ["Level", "Aspect", "Spearman", "Pearson", "Kendall"]
        table_data = []
        
        for aspect in ["document", "interaction"]:
            if aspect in export_data["results"]:
                results = export_data["results"][aspect]
                
                # Instance level
                if results["instance_level"]["spearman"] is not None:
                    table_data.append([
                        "Instance",
                        aspect.title(),
                        f"{results['instance_level']['spearman']:.3f}",
                        f"{results['instance_level']['pearson']:.3f}",
                        f"{results['instance_level']['kendall']:.3f}"
                    ])
                
                # Intermediate level
                if results["intermediate_level"]["spearman"] is not None:
                    table_data.append([
                        "Intermediate",
                        aspect.title(),
                        f"{results['intermediate_level']['spearman']:.3f}",
                        f"{results['intermediate_level']['pearson']:.3f}",
                        f"{results['intermediate_level']['kendall']:.3f}"
                    ])
                
                # System level
                if results["system_level"]["spearman"] is not None:
                    table_data.append([
                        "System",
                        aspect.title(),
                        f"{results['system_level']['spearman']:.3f}",
                        f"{results['system_level']['pearson']:.3f}",
                        f"{results['system_level']['kendall']:.3f}"
                    ])
        
        if table_data:
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        print("="*80)

if __name__ == "__main__":
    main()