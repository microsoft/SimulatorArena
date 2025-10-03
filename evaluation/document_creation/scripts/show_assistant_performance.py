#!/usr/bin/env python3
"""
Show assistant performance metrics for document creation task.
This script displays document quality and interaction quality ratings for each assistant model.

Unlike math tutoring, document creation uses ratings for both aspects (no correctness/F1).
"""

import json
import argparse
import statistics
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tabulate import tabulate

def load_evaluation_data(
    file_name: str,
    annotation_id: str = "document_creation_annotations"
) -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict]]:
    """
    Load evaluation data for document and interaction ratings, plus terminated conversations.
    
    Returns:
        Tuple of (document_ratings, interaction_ratings, terminated_conversations)
    """
    # Base evaluation directory
    eval_dir = Path(__file__).parent.parent / "evaluation_outputs"
    
    # Load document ratings
    document_file = eval_dir / "document_rating" / f"{file_name}.json"
    document_ratings = None
    if document_file.exists():
        with open(document_file, 'r') as f:
            data = json.load(f)
            # Handle both direct format and wrapped format
            if "evaluations" in data:
                document_ratings = data["evaluations"]
            else:
                document_ratings = data
    
    # Load interaction ratings
    interaction_file = eval_dir / "interaction_rating" / f"{file_name}.json"
    interaction_ratings = None
    if interaction_file.exists():
        with open(interaction_file, 'r') as f:
            data = json.load(f)
            # Handle both direct format and wrapped format
            if "evaluations" in data:
                interaction_ratings = data["evaluations"]
            else:
                interaction_ratings = data
    
    # Load terminated conversations for turn counts
    terminated_file = Path(__file__).parent.parent.parent.parent / "simulation" / "terminated_conversations" / annotation_id / f"{file_name}.json"
    terminated_conversations = None
    if terminated_file.exists():
        with open(terminated_file, 'r') as f:
            terminated_conversations = json.load(f)
    
    return document_ratings, interaction_ratings, terminated_conversations

def calculate_model_statistics(
    document_ratings: Dict,
    interaction_ratings: Dict,
    terminated_conversations: Dict
) -> List[Tuple]:
    """
    Calculate statistics for each model.
    
    Returns:
        List of tuples: (model_name, avg_doc_rating, std_doc_rating, 
                        avg_int_rating, std_int_rating, avg_turns, n_conversations)
    """
    model_stats = []
    
    # Get all models from the data
    models = set()
    if document_ratings:
        models.update(document_ratings.keys())
    if interaction_ratings:
        models.update(interaction_ratings.keys())
    if terminated_conversations:
        models.update(terminated_conversations.keys())
    
    for model in sorted(models):
        doc_ratings = []
        int_ratings = []
        turn_counts = []
        
        # Collect document ratings
        if document_ratings and model in document_ratings:
            for doc_type in document_ratings[model]:
                for intent in document_ratings[model][doc_type]:
                    for worker_id, rating_data in document_ratings[model][doc_type][intent].items():
                        try:
                            # Handle different formats of rating storage
                            if isinstance(rating_data, dict):
                                if "extracted_rating" in rating_data:
                                    rating = float(rating_data["extracted_rating"])
                                elif "rating" in rating_data:
                                    rating = float(rating_data["rating"])
                                else:
                                    continue
                            else:
                                rating = float(rating_data)
                            
                            if 1 <= rating <= 10:
                                doc_ratings.append(rating)
                        except (ValueError, TypeError, KeyError):
                            continue
        
        # Collect interaction ratings
        if interaction_ratings and model in interaction_ratings:
            for doc_type in interaction_ratings[model]:
                for intent in interaction_ratings[model][doc_type]:
                    for worker_id, rating_data in interaction_ratings[model][doc_type][intent].items():
                        try:
                            # Handle different formats of rating storage
                            if isinstance(rating_data, dict):
                                if "extracted_rating" in rating_data:
                                    rating = float(rating_data["extracted_rating"])
                                elif "rating" in rating_data:
                                    rating = float(rating_data["rating"])
                                else:
                                    continue
                            else:
                                rating = float(rating_data)
                            
                            if 1 <= rating <= 10:
                                int_ratings.append(rating)
                        except (ValueError, TypeError, KeyError):
                            continue
        
        # Collect turn counts
        if terminated_conversations and model in terminated_conversations:
            for doc_type in terminated_conversations[model]:
                for intent in terminated_conversations[model][doc_type]:
                    for worker_id, term_data in terminated_conversations[model][doc_type][intent].items():
                        if "ending_turn_number" in term_data:
                            turn_counts.append(term_data["ending_turn_number"])
        
        # Calculate statistics
        if doc_ratings or int_ratings:
            avg_doc = statistics.mean(doc_ratings) if doc_ratings else None
            std_doc = statistics.stdev(doc_ratings) if len(doc_ratings) > 1 else 0.0
            
            avg_int = statistics.mean(int_ratings) if int_ratings else None
            std_int = statistics.stdev(int_ratings) if len(int_ratings) > 1 else 0.0
            
            avg_turns = statistics.mean(turn_counts) if turn_counts else None
            
            # Use the maximum count as the number of conversations
            n_conversations = max(len(doc_ratings), len(int_ratings))
            
            model_stats.append((
                model,
                avg_doc,
                std_doc,
                avg_int,
                std_int,
                avg_turns,
                n_conversations
            ))
    
    return model_stats

def display_performance_table(
    model_stats: List[Tuple],
    sort_by: str = "document",
    top_k: Optional[int] = None
):
    """
    Display performance table for assistant models.
    
    Args:
        model_stats: List of model statistics tuples
        sort_by: Sort criterion ('document', 'interaction', 'combined')
        top_k: If specified, only show top k models
    """
    # Sort based on criterion
    if sort_by == "document":
        model_stats.sort(key=lambda x: x[1] if x[1] is not None else 0, reverse=True)
    elif sort_by == "interaction":
        model_stats.sort(key=lambda x: x[3] if x[3] is not None else 0, reverse=True)
    elif sort_by == "combined":
        # Sort by average of document and interaction ratings
        def combined_score(x):
            doc = x[1] if x[1] is not None else 0
            int_r = x[3] if x[3] is not None else 0
            count = sum([1 for v in [x[1], x[3]] if v is not None])
            return (doc + int_r) / count if count > 0 else 0
        model_stats.sort(key=combined_score, reverse=True)
    
    # Limit to top k if specified
    if top_k:
        model_stats = model_stats[:top_k]
    
    # Prepare table data
    headers = [
        "Rank",
        "Model",
        "Doc Rating",
        "Doc Std",
        "Int Rating", 
        "Int Std",
        "Avg Turns",
        "N"
    ]
    
    table_data = []
    for rank, (model, avg_doc, std_doc, avg_int, std_int, avg_turns, n) in enumerate(model_stats, 1):
        # Format the values
        doc_str = f"{avg_doc:.2f}" if avg_doc is not None else "N/A"
        doc_std_str = f"±{std_doc:.2f}" if avg_doc is not None else ""
        int_str = f"{avg_int:.2f}" if avg_int is not None else "N/A"
        int_std_str = f"±{std_int:.2f}" if avg_int is not None else ""
        turns_str = f"{avg_turns:.1f}" if avg_turns is not None else "N/A"
        
        table_data.append([
            rank,
            model,
            doc_str,
            doc_std_str,
            int_str,
            int_std_str,
            turns_str,
            n
        ])
    
    print("\n" + "="*80)
    print("ASSISTANT PERFORMANCE - DOCUMENT CREATION TASK")
    print("="*80)
    print(f"Sorted by: {sort_by.title()} Rating")
    print("-"*80)
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Calculate and display overall statistics
    all_doc_ratings = [x[1] for x in model_stats if x[1] is not None]
    all_int_ratings = [x[3] for x in model_stats if x[3] is not None]
    all_turns = [x[5] for x in model_stats if x[5] is not None]
    
    print("\nOVERALL STATISTICS:")
    print("-"*40)
    if all_doc_ratings:
        print(f"Document Ratings:")
        print(f"  Mean: {statistics.mean(all_doc_ratings):.2f}")
        print(f"  Std:  {statistics.stdev(all_doc_ratings):.2f}" if len(all_doc_ratings) > 1 else "  Std:  N/A")
        print(f"  Min:  {min(all_doc_ratings):.2f}")
        print(f"  Max:  {max(all_doc_ratings):.2f}")
    
    if all_int_ratings:
        print(f"\nInteraction Ratings:")
        print(f"  Mean: {statistics.mean(all_int_ratings):.2f}")
        print(f"  Std:  {statistics.stdev(all_int_ratings):.2f}" if len(all_int_ratings) > 1 else "  Std:  N/A")
        print(f"  Min:  {min(all_int_ratings):.2f}")
        print(f"  Max:  {max(all_int_ratings):.2f}")
    
    if all_turns:
        print(f"\nConversation Turns:")
        print(f"  Mean: {statistics.mean(all_turns):.1f}")
        print(f"  Min:  {min(all_turns):.1f}")
        print(f"  Max:  {max(all_turns):.1f}")
    
    print("="*80)

def main():
    parser = argparse.ArgumentParser(
        description="Display assistant performance metrics for document creation task."
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
        help="Annotation dataset ID"
    )
    parser.add_argument(
        "--sort_by",
        type=str,
        choices=["document", "interaction", "combined"],
        default="document",
        help="Sort models by this criterion (default: document)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Show only top k models"
    )
    parser.add_argument(
        "--export",
        type=str,
        help="Export results to JSON file"
    )
    
    args = parser.parse_args()
    
    # Load evaluation data
    print(f"Loading evaluation data for: {args.file_name}")
    document_ratings, interaction_ratings, terminated_conversations = load_evaluation_data(
        args.file_name,
        args.annotation_id
    )
    
    if not document_ratings and not interaction_ratings:
        print("ERROR: No evaluation data found.")
        print("Please ensure evaluation has been completed for this file.")
        return
    
    # Calculate statistics
    model_stats = calculate_model_statistics(
        document_ratings or {},
        interaction_ratings or {},
        terminated_conversations or {}
    )
    
    if not model_stats:
        print("No valid statistics could be calculated from the data.")
        return
    
    # Display results
    display_performance_table(
        model_stats,
        sort_by=args.sort_by,
        top_k=args.top_k
    )
    
    # Export if requested
    if args.export:
        export_data = {
            "file_name": args.file_name,
            "annotation_id": args.annotation_id,
            "sort_by": args.sort_by,
            "models": []
        }
        
        for model, avg_doc, std_doc, avg_int, std_int, avg_turns, n in model_stats:
            export_data["models"].append({
                "model": model,
                "document_rating": {
                    "mean": avg_doc,
                    "std": std_doc
                },
                "interaction_rating": {
                    "mean": avg_int,
                    "std": std_int
                },
                "avg_turns": avg_turns,
                "n_conversations": n
            })
        
        with open(args.export, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"\nResults exported to: {args.export}")

if __name__ == "__main__":
    main()