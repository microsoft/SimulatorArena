#!/usr/bin/env python3
"""
Process batch evaluation using OpenAI's batch API.
This script submits batch evaluation requests and retrieves results.
Based on the existing notebook implementations.

Key features:
- Checks for existing batches before creating duplicates
- Configurable polling interval for result retrieval
- Auto-resume capability for interrupted processes
- Comprehensive status reporting
"""

import os
import json
import time
import argparse
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from openai import OpenAI
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

def extract_rating(text: str) -> str:
    """
    Extract the numeric rating (1-10) from the given text.
    Based on retrieve_batch_results_extrinsic_evaluation.ipynb
    """
    tail = text[text.rfind("Rating") + len("Rating"):] if "Rating" in text else text

    # Quick half-point checks
    for half in ("9.5", "8.5", "7.5"):
        if half in tail:
            return half

    # Clean markdown formatting
    clean_text = re.sub(r'[*#_`]', '', text)

    # Strategy 1: Number on the next non-empty line
    pattern_next_line = re.compile(r'\bRating\b[^\n]*\n(?:[ \t]*\n)*[ \t]*(\d+)(?:\s*/\s*10)?')
    last_match_val = None
    for match in pattern_next_line.finditer(clean_text):
        rating_val = int(match.group(1))
        if 1 <= rating_val <= 10:
            last_match_val = str(rating_val)
            
    if last_match_val:
        return last_match_val

    # Strategy 2: Number on the same line
    pattern_same_line = re.compile(r'\bRating\b\s*:?\s*(\d+)(?:\s*/\s*10)?')
    last_match_val = None
    for match in pattern_same_line.finditer(clean_text):
        rating_val = int(match.group(1))
        if 1 <= rating_val <= 10:
            last_match_val = str(rating_val)

    if last_match_val:
        return last_match_val

    # Strategy 3: Line split fallback
    lines = clean_text.split('\n')
    rating_line_index = -1
    for i, line in enumerate(lines):
        if re.search(r'\bRating\b', line):
            rating_line_index = i

    if rating_line_index != -1:
        for j in range(rating_line_index + 1, min(rating_line_index + 4, len(lines))):
            next_line = lines[j].strip()
            if next_line:
                num_match = re.match(r'(\d+)(?:\s*/\s*10)?', next_line)
                if num_match and 1 <= int(num_match.group(1)) <= 10:
                    return num_match.group(1)
                break
    
    # Fallback search
    for num in ["10", "9", "8", "7", "6", "5", "4", "3", "2", "1"]:
        if num in tail:
            return num
    
    return "Error"

def extract_student_answer(text: str) -> str:
    """
    Extract the student's answer from the provided text.
    Based on retrieve_batch_results_extract_answer.ipynb
    """
    match = re.search(r'## Extracted Student\'s Answer:\s*(.*)', text, re.DOTALL)
    if match:
        answer = match.group(1).strip()
        return answer if answer else "Error"
    return "Error"

def get_ending_turn_number(text: str) -> Optional[int]:
    """
    Extract the ending turn number from the generated text.
    Based on retrieve_batch_results_terminate_conversation.ipynb
    """
    pattern = r'"Ending Turn Number"\s*:\s*(\d+)'
    match = re.search(pattern, text)
    if match:
        return int(match.group(1))
    return None

def extract_correctness(text: str) -> str:
    """
    Extract correctness from evaluation output.
    Based on retrieve_batch_results_check_correctness.ipynb
    """
    # Check the last 20 characters for correctness indicator
    tail = text[-20:].lower()
    if "incorrect" in tail:
        return "incorrect"
    elif "correct" in tail:
        return "correct"
    else:
        return "Error"

def merge_nested_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """
    Merges two nested dictionaries with unique ending keys.
    """
    result = dict1.copy()
    
    def recursive_merge(current_dict, other_dict):
        for key, value in other_dict.items():
            if key not in current_dict:
                current_dict[key] = value
            elif isinstance(value, dict) and isinstance(current_dict[key], dict):
                recursive_merge(current_dict[key], value)
            else:
                print(f"Conflict at key: {key}. Keeping existing value.")
                continue
    
    recursive_merge(result, dict2)
    return result

class BatchEvaluationProcessor:
    """Handles batch evaluation processing using OpenAI's batch API."""
    
    def __init__(self, api_key: str = None):
        """Initialize the batch processor."""
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
    
    def check_existing_batch(self, batch_data_path: str) -> Optional[Dict]:
        """
        Check if a batch has already been submitted for this file.
        
        Returns:
            Dict with batch info if exists, None otherwise
        """
        # Check for log file
        batch_path = Path(batch_data_path)
        log_path = batch_path.parent / "logs" / batch_path.stem
        log_file = log_path.with_suffix(".json")
        
        if log_file.exists():
            print(f"Found existing batch log: {log_file}")
            with open(log_file, 'r') as f:
                log_data = json.load(f)
            
            batch_id = log_data.get("message_batch", {}).get("id")
            if batch_id:
                # Check batch status
                try:
                    batch = self.client.batches.retrieve(batch_id)
                    print(f"Existing batch found: {batch_id}")
                    print(f"Status: {batch.status}")
                    
                    return {
                        "batch_id": batch_id,
                        "log_file": str(log_file),
                        "status": batch.status,
                        "batch": batch
                    }
                except Exception as e:
                    print(f"Could not retrieve batch {batch_id}: {e}")
                    print("Will create a new batch.")
        
        return None
    
    def process_batch_file(
        self,
        batch_data_path: str,
        model: str = "gpt-4o-2024-11-20",
        description: str = None,
        wait_for_completion: bool = False,
        polling_interval: int = 60,
        check_existing: bool = True
    ) -> Dict[str, Any]:
        """
        Process a batch evaluation file following the notebook workflow.
        
        Args:
            batch_data_path: Path to batch data JSON file
            model: Model to use for evaluation
            description: Description for the batch
            wait_for_completion: Whether to wait for batch to complete
            polling_interval: Seconds between status checks (default 60)
            check_existing: Whether to check for existing batch first
        """
        # Check for existing batch
        if check_existing:
            existing = self.check_existing_batch(batch_data_path)
            if existing:
                if existing["status"] == "completed":
                    print("Batch already completed! Retrieving results...")
                    return self.retrieve_results(existing["batch_id"], existing["log_file"])
                elif existing["status"] in ["failed", "expired", "cancelled"]:
                    print(f"Previous batch {existing['status']}. Creating new batch...")
                elif existing["status"] in ["validating", "in_progress", "finalizing"]:
                    if wait_for_completion:
                        print(f"Batch is {existing['status']}. Waiting for completion...")
                        return self.wait_for_batch(
                            existing["batch_id"], 
                            existing["log_file"],
                            polling_interval
                        )
                    else:
                        return {
                            "batch_id": existing["batch_id"],
                            "log_file": existing["log_file"],
                            "status": existing["status"],
                            "message": f"Batch already in progress. Use --wait to wait for completion."
                        }
        
        # Load batch data
        with open(batch_data_path, 'r') as f:
            batch_data = json.load(f)
        
        contexts = batch_data["contexts"]
        keys = batch_data["keys"]
        metadata = batch_data.get("metadata", {})
        
        # Determine model based on task type
        if "terminated_conversations" in batch_data_path:
            model = "gpt-4o-2024-05-13"
        
        # Create JSONL file path
        folder_path = Path(batch_data_path).parent
        file_name = Path(batch_data_path).stem
        jsonl_folder = folder_path / "openai_jsonl"
        jsonl_folder.mkdir(parents=True, exist_ok=True)
        jsonl_path = jsonl_folder / f"{file_name}.jsonl"
        
        print(f"Creating JSONL file at: {jsonl_path}")
        
        # Create JSONL file
        requests = []
        key_dict = {}
        
        for i, (key, context) in enumerate(zip(keys, contexts)):
            batch_request = {
                "custom_id": str(i),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": context,
                    "temperature": 0.7,
                    "top_p": 1.0,
                    "max_tokens": 4000
                }
            }
            key_dict[str(i)] = key
            requests.append(batch_request)
        
        with open(jsonl_path, "w") as f:
            for r in requests:
                json.dump(r, f)
                f.write("\n")
        
        print(f"Created JSONL with {len(requests)} requests")
        
        # Upload file
        print("Uploading file to OpenAI...")
        try:
            batch_input_file = self.client.files.create(
                file=open(jsonl_path, "rb"),
                purpose="batch"
            )
            print(f"File uploaded: {batch_input_file.id}")
        except Exception as e:
            print(f"Error uploading file: {e}")
            return {"error": f"Failed to upload file: {e}"}
        
        # Create batch
        print("Creating batch...")
        try:
            batch_result = self.client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                    "description": description or metadata.get("evaluation_type", "evaluation")
                }
            )
            print(f"Batch created: {batch_result.id}")
        except Exception as e:
            print(f"Error creating batch: {e}")
            return {"error": f"Failed to create batch: {e}"}
        
        # Save batch info to logs
        log_save_path = folder_path / "logs" / file_name
        log_save_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_save_path.with_suffix(".json")
        
        saving_dict = {
            "message_batch": batch_result.to_dict(),
            "key_dict": key_dict,
            "metadata": metadata,
            "created_at": datetime.now().isoformat(),
            "batch_data_path": batch_data_path,
            "model": model
        }
        
        with open(log_file, 'w') as f:
            json.dump(saving_dict, f, indent=4, default=str)
        
        print(f"Batch info saved to: {log_file}")
        
        if wait_for_completion:
            return self.wait_for_batch(batch_result.id, str(log_file), polling_interval)
        
        return {"batch_id": batch_result.id, "log_file": str(log_file)}
    
    def wait_for_batch(
        self, 
        batch_id: str, 
        log_file: str,
        polling_interval: int = 60
    ) -> Dict[str, Any]:
        """
        Wait for batch to complete with configurable polling.
        
        Args:
            batch_id: Batch ID to wait for
            log_file: Path to log file
            polling_interval: Seconds between checks (default 60)
        """
        print(f"Waiting for batch to complete (checking every {polling_interval} seconds)...")
        start_time = time.time()
        last_status = None
        
        while True:
            try:
                batch = self.client.batches.retrieve(batch_id)
                
                # Only print status if it changed
                if batch.status != last_status:
                    elapsed = int(time.time() - start_time)
                    print(f"[{elapsed}s] Status: {batch.status}")
                    
                    # Print progress if available
                    if hasattr(batch, 'request_counts') and batch.request_counts:
                        counts = batch.request_counts
                        if hasattr(counts, 'completed') and hasattr(counts, 'total'):
                            if counts.total > 0:
                                progress = (counts.completed / counts.total) * 100
                                print(f"  Progress: {counts.completed}/{counts.total} ({progress:.1f}%)")
                    
                    last_status = batch.status
                
                if batch.status == "completed":
                    print(f"Batch completed in {int(time.time() - start_time)} seconds!")
                    return self.retrieve_results(batch_id, log_file)
                    
                elif batch.status in ["failed", "expired", "cancelled"]:
                    error_msg = f"Batch {batch.status}"
                    if hasattr(batch, 'errors') and batch.errors:
                        error_msg += f": {batch.errors}"
                    print(f"Batch failed: {error_msg}")
                    return {"error": error_msg}
                
            except Exception as e:
                print(f"Error checking batch status: {e}")
                print("Will retry in next interval...")
            
            time.sleep(polling_interval)
    
    def retrieve_results(
        self,
        batch_id: str = None,
        log_file_path: str = None
    ) -> Dict[str, Any]:
        """
        Retrieve and process batch results based on evaluation type.
        """
        # Load log file to get keys and metadata
        if log_file_path:
            with open(log_file_path, 'r') as f:
                log_data = json.load(f)
        else:
            raise ValueError("log_file_path is required to retrieve results")
        
        key_dict = log_data["key_dict"]
        metadata = log_data.get("metadata", {})
        
        if not batch_id:
            batch_id = log_data["message_batch"]["id"]
        
        # Retrieve batch
        batch = self.client.batches.retrieve(batch_id)
        
        if batch.status != "completed":
            return {"error": f"Batch not completed. Status: {batch.status}"}
        
        print(f"Retrieving results for batch: {batch_id}")
        
        # Get output file
        try:
            file_response = self.client.files.content(batch.output_file_id)
        except Exception as e:
            return {"error": f"Failed to retrieve output file: {e}"}
        
        # Process results based on evaluation type
        evaluation_type = metadata.get("evaluation_type", "")
        
        if "interaction_rating" in evaluation_type or "extrinsic_evaluation" in str(log_file_path):
            return self._process_rating_results(file_response.text, key_dict, metadata)
        elif "answer_extraction" in evaluation_type or "extract" in str(log_file_path):
            return self._process_answer_extraction_results(file_response.text, key_dict, metadata)
        elif "correctness" in evaluation_type or "check_correctness" in str(log_file_path):
            return self._process_correctness_results(file_response.text, key_dict, metadata)
        elif "terminate" in str(log_file_path):
            return self._process_termination_results(file_response.text, key_dict, metadata)
        else:
            return self._process_generic_results(file_response.text, key_dict)
    
    def _process_rating_results(self, response_text: str, key_dict: Dict, metadata: Dict = None) -> Dict:
        """Process interaction rating results."""
        result_dict = {}
        errors = []
        
        for line in response_text.strip().splitlines():
            if not line:
                continue
                
            result = json.loads(line)
            custom_id = result['custom_id']
            
            if result.get("error"):
                errors.append(f"Error for custom_id {custom_id}: {result['error']}")
                continue
            
            result_key = key_dict[custom_id]
            output = result["response"]["body"]["choices"][0]["message"]["content"]
            
            if not output:
                errors.append(f"Empty output for: {custom_id}")
                continue
            
            # Extract rating
            rating = extract_rating(output)
            if rating == "Error":
                errors.append(f"Error extracting rating for response {custom_id}")
                continue
            
            # Organize by model, problem, user (exact same format as original)
            model_name, problem_id, user_key = result_key
            
            if model_name not in result_dict:
                result_dict[model_name] = {}
            if problem_id not in result_dict[model_name]:
                result_dict[model_name][problem_id] = {}
            
            result_dict[model_name][problem_id][user_key] = {
                "extracted_rating": rating,
                "output": output
            }
        
        # Print summary
        total = sum(len(users) for model in result_dict.values() for users in model.values())
        print(f"Successfully processed {total} ratings")
        if errors:
            print(f"Encountered {len(errors)} errors")
        
        # Return with evaluations wrapper for consistency with new structure
        # But the content inside is identical to original format
        return {"evaluations": result_dict, "errors": errors if errors else None}
    
    def _process_answer_extraction_results(self, response_text: str, key_dict: Dict, metadata: Dict = None) -> Dict:
        """Process answer extraction results."""
        result_dict = {}
        errors = []
        
        for line in response_text.strip().splitlines():
            if not line:
                continue
                
            result = json.loads(line)
            custom_id = result['custom_id']
            
            if result.get("error"):
                errors.append(f"Error for custom_id {custom_id}: {result['error']}")
                continue
            
            result_key = key_dict[custom_id]
            output = result["response"]["body"]["choices"][0]["message"]["content"]
            
            if not output:
                errors.append(f"Empty output for: {custom_id}")
                continue
            
            # Extract answer
            extracted = extract_student_answer(output)
            if extracted == "Error":
                errors.append(f"Error extracting answer at index {custom_id}")
                continue
            
            # Organize results
            model_name, problem_id, user_key = result_key
            
            if model_name not in result_dict:
                result_dict[model_name] = {}
            if problem_id not in result_dict[model_name]:
                result_dict[model_name][problem_id] = {}
            
            result_dict[model_name][problem_id][user_key] = {
                "extracted_answer": extracted,
                "output": output
            }
        
        # Print summary
        total = sum(len(users) for model in result_dict.values() for users in model.values())
        print(f"Successfully extracted {total} answers")
        if errors:
            print(f"Encountered {len(errors)} errors")
        
        return {"answers": result_dict, "errors": errors if errors else None}
    
    def _process_correctness_results(self, response_text: str, key_dict: Dict, metadata: Dict = None) -> Dict:
        """Process correctness evaluation results."""
        result_dict = {}
        errors = []
        
        for line in response_text.strip().splitlines():
            if not line:
                continue
                
            result = json.loads(line)
            custom_id = result['custom_id']
            
            if result.get("error"):
                errors.append(f"Error for custom_id {custom_id}: {result['error']}")
                continue
            
            result_key = key_dict[custom_id]
            output = result["response"]["body"]["choices"][0]["message"]["content"]
            
            if not output:
                errors.append(f"Empty output for: {custom_id}")
                continue
            
            # Extract correctness
            correctness = extract_correctness(output)
            if correctness == "Error":
                errors.append(f"Could not determine correctness for response {custom_id}")
                continue
            
            # Organize results
            model_name, problem_id, user_key = result_key
            
            if model_name not in result_dict:
                result_dict[model_name] = {}
            if problem_id not in result_dict[model_name]:
                result_dict[model_name][problem_id] = {}
            
            result_dict[model_name][problem_id][user_key] = {
                "correctness": correctness,
                "correctness_analysis": output
            }
        
        # Print summary
        total = sum(len(users) for model in result_dict.values() for users in model.values())
        correct = sum(1 for model in result_dict.values() for problem in model.values() 
                      for user in problem.values() if user.get("correctness") == "correct")
        print(f"Successfully evaluated {total} answers")
        print(f"Correct: {correct}/{total} ({correct/total*100:.1f}%)")
        if errors:
            print(f"Encountered {len(errors)} errors")
        
        return {"correctness": result_dict, "errors": errors if errors else None}
    
    def _process_termination_results(self, response_text: str, key_dict: Dict, metadata: Dict = None) -> Dict:
        """Process conversation termination results."""
        result_dict = {}
        errors = []
        
        for line in response_text.strip().splitlines():
            if not line:
                continue
                
            result = json.loads(line)
            custom_id = result['custom_id']
            
            if result.get("error"):
                errors.append(f"Error for custom_id {custom_id}: {result['error']}")
                continue
            
            result_key = key_dict[custom_id]
            output = result["response"]["body"]["choices"][0]["message"]["content"]
            
            if not output:
                errors.append(f"Empty output for: {custom_id}")
                continue
            
            # Extract ending turn number
            number = get_ending_turn_number(output)
            if not number:
                errors.append(f"Failed to extract ending turn number from response {custom_id}")
                continue
            
            # Organize results
            model_name, problem_id, user_key = result_key
            
            if model_name not in result_dict:
                result_dict[model_name] = {}
            if problem_id not in result_dict[model_name]:
                result_dict[model_name][problem_id] = {}
            
            result_dict[model_name][problem_id][user_key] = {
                "output": output,
                "ending_turn_number": number
            }
        
        # Print summary
        total = sum(len(users) for model in result_dict.values() for users in model.values())
        print(f"Successfully processed {total} termination points")
        if errors:
            print(f"Encountered {len(errors)} errors")
        
        return {"terminations": result_dict, "errors": errors if errors else None}
    
    def _process_generic_results(self, response_text: str, key_dict: Dict) -> Dict:
        """Process generic results."""
        results = []
        errors = []
        
        for line in response_text.strip().splitlines():
            if not line:
                continue
                
            result = json.loads(line)
            custom_id = result['custom_id']
            
            if result.get("error"):
                errors.append(f"Error for custom_id {custom_id}: {result['error']}")
                continue
            
            result_key = key_dict[custom_id]
            output = result["response"]["body"]["choices"][0]["message"]["content"]
            
            results.append({
                "key": result_key,
                "output": output
            })
        
        print(f"Successfully processed {len(results)} results")
        if errors:
            print(f"Encountered {len(errors)} errors")
        
        return {"results": results, "errors": errors if errors else None}

def main():
    parser = argparse.ArgumentParser(
        description="Process batch evaluation using OpenAI's batch API"
    )
    parser.add_argument(
        "--batch_file",
        type=str,
        required=True,
        help="Path to batch data JSON file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to save evaluation results"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-2024-11-20",
        help="Model to use for evaluation"
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for batch to complete and retrieve results"
    )
    parser.add_argument(
        "--polling_interval",
        type=int,
        default=60,
        help="Seconds between status checks when waiting (default: 60)"
    )
    parser.add_argument(
        "--batch_id",
        type=str,
        help="Existing batch ID to check status or retrieve results"
    )
    parser.add_argument(
        "--log_file",
        type=str,
        help="Path to log file for existing batch"
    )
    parser.add_argument(
        "--no_check_existing",
        action="store_true",
        help="Don't check for existing batch, always create new"
    )
    
    args = parser.parse_args()
    
    processor = BatchEvaluationProcessor()
    
    if args.batch_id:
        # Retrieve results for existing batch
        if not args.log_file:
            # Try to find log file
            batch_path = Path(args.batch_file)
            log_path = batch_path.parent / "logs" / batch_path.stem
            log_file = log_path.with_suffix(".json")
            if log_file.exists():
                args.log_file = str(log_file)
            else:
                print(f"Error: Could not find log file. Please provide --log_file")
                return
        
        if args.wait:
            # Wait for completion if not already complete
            results = processor.wait_for_batch(args.batch_id, args.log_file, args.polling_interval)
        else:
            # Just retrieve results
            results = processor.retrieve_results(args.batch_id, args.log_file)
        
        # Handle errors
        if "error" in results:
            print(f"Error: {results['error']}")
            return
        
        # Save results
        if args.output_file:
            output_path = args.output_file
        else:
            # Create output path based on evaluation type
            batch_path = Path(args.batch_file)
            
            # Determine output location based on evaluation type
            if "interaction_rating" in str(batch_path):
                output_dir = batch_path.parent.parent / "evaluation_outputs" / "interaction_rating"
            elif "answer_extraction" in str(batch_path):
                output_dir = batch_path.parent.parent / "evaluation_outputs" / "extracted_answer"
            elif "correctness" in str(batch_path) or "check_correctness" in str(batch_path):
                # For correctness, we'll update the extracted_answer file directly
                output_dir = batch_path.parent.parent / "evaluation_outputs" / "extracted_answer"
            else:
                output_dir = batch_path.parent.parent / "evaluation_outputs"
            
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / batch_path.name
        
        # Load existing results if present
        if Path(output_path).exists():
            with open(output_path, 'r') as f:
                existing = json.load(f)
            
            # Merge results based on type
            if "evaluations" in results:
                existing_evals = existing.get("evaluations", {})
                results["evaluations"] = merge_nested_dicts(existing_evals, results["evaluations"])
            elif "answers" in results:
                existing_answers = existing.get("answers", {})
                results["answers"] = merge_nested_dicts(existing_answers, results["answers"])
            elif "correctness" in results:
                # For correctness, merge with existing extracted answers
                if "answers" in existing:
                    # Update existing answers with correctness data
                    for model_name, model_data in results["correctness"].items():
                        if model_name not in existing["answers"]:
                            existing["answers"][model_name] = {}
                        for problem_id, problem_data in model_data.items():
                            if problem_id not in existing["answers"][model_name]:
                                existing["answers"][model_name][problem_id] = {}
                            for user_key, correctness_data in problem_data.items():
                                if user_key not in existing["answers"][model_name][problem_id]:
                                    existing["answers"][model_name][problem_id][user_key] = {}
                                # Add correctness fields to existing answer data
                                existing["answers"][model_name][problem_id][user_key].update(correctness_data)
                    results = existing
                else:
                    # If no existing answers, save correctness results separately (shouldn't happen)
                    existing_correct = existing.get("correctness", {})
                    results["correctness"] = merge_nested_dicts(existing_correct, results["correctness"])
            elif "terminations" in results:
                existing_terms = existing.get("terminations", {})
                results["terminations"] = merge_nested_dicts(existing_terms, results["terminations"])
        
        # Save merged results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"Results saved to: {output_path}")
        
    else:
        # Process batch file (new or existing)
        result = processor.process_batch_file(
            args.batch_file,
            model=args.model,
            wait_for_completion=args.wait,
            polling_interval=args.polling_interval,
            check_existing=not args.no_check_existing
        )
        
        if "error" in result:
            print(f"Error: {result['error']}")
        elif "message" in result:
            print(f"\n{result['message']}")
            print(f"Batch ID: {result['batch_id']}")
            print(f"Log file: {result['log_file']}")
        elif "evaluations" in result or "answers" in result or "correctness" in result:
            # Results were retrieved (batch was already complete or we waited)
            print("\nBatch processing complete!")
            
            # Save results
            if args.output_file:
                output_path = args.output_file
            else:
                batch_path = Path(args.batch_file)
                
                # Determine output location based on evaluation type
                if "interaction_rating" in str(batch_path):
                    output_dir = batch_path.parent.parent / "evaluation_outputs" / "interaction_rating"
                elif "answer_extraction" in str(batch_path):
                    output_dir = batch_path.parent.parent / "evaluation_outputs" / "extracted_answer"
                elif "correctness" in str(batch_path) or "check_correctness" in str(batch_path):
                    # For correctness, we'll update the extracted_answer file directly
                    output_dir = batch_path.parent.parent / "evaluation_outputs" / "extracted_answer"
                else:
                    output_dir = batch_path.parent.parent / "evaluation_outputs"
                
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / batch_path.name
            
            # Load existing results for correctness updates
            if ("correctness" in result or "check_correctness" in str(batch_path)) and Path(output_path).exists():
                with open(output_path, 'r') as f:
                    existing = json.load(f)
                
                # Update existing answers with correctness data
                if "correctness" in result and "answers" in existing:
                    for model_name, model_data in result["correctness"].items():
                        if model_name not in existing["answers"]:
                            existing["answers"][model_name] = {}
                        for problem_id, problem_data in model_data.items():
                            if problem_id not in existing["answers"][model_name]:
                                existing["answers"][model_name][problem_id] = {}
                            for user_key, correctness_data in problem_data.items():
                                if user_key not in existing["answers"][model_name][problem_id]:
                                    existing["answers"][model_name][problem_id][user_key] = {}
                                existing["answers"][model_name][problem_id][user_key].update(correctness_data)
                    result = existing
            
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=4)
            
            print(f"Results saved to: {output_path}")
        elif "batch_id" in result:
            print(f"\nBatch submitted successfully!")
            print(f"Batch ID: {result['batch_id']}")
            print(f"Log file: {result['log_file']}")
            print(f"\nTo check status and retrieve results later, run:")
            print(f"  python {__file__} --batch_file {args.batch_file} --batch_id {result['batch_id']} --wait")

if __name__ == "__main__":
    main()