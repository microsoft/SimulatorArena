#!/usr/bin/env python3
"""
Process batch evaluation for document creation using OpenAI's batch API.
Handles both rating evaluations (document/interaction) and document extraction.

This script:
1. Submits batch evaluation requests to OpenAI
2. Monitors batch status
3. Retrieves and processes results
4. Saves structured output for analysis

Based on document_creation/user_simulation/batch_prompts notebooks.
"""

import os
import json
import time
import argparse
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from openai import OpenAI
import sys

from dotenv import load_dotenv
dotenv_path = os.path.expanduser('~/.env')
load_dotenv(dotenv_path)
os.environ['CURL_CA_BUNDLE'] = ''

def extract_rating(text: str) -> str:
    """
    Extract the numeric rating (1-10) from evaluation text.
    Based on retrieve_batch_results_extrinsic_evaluation_rating_openai.ipynb
    
    Handles formats like:
    - "Rating: 8"
    - "**Rating**: **9**" (after markdown removal)
    - "##### Rating\n7/10" (after markdown removal)
    """
    tail = text[text.rfind("Rating") + len("Rating"):] if "Rating" in text else text

    # Quick half-point checks
    for half in ("9.5", "8.5", "7.5", "6.5", "5.5", "4.5", "3.5", "2.5", "1.5"):
        if half in tail:
            return half

    # Clean markdown formatting
    clean_text = re.sub(r'[*#_`]', '', text)

    # Strategy 1: Number on the next non-empty line
    pattern_next_line = re.compile(r'\bRating\b[^\n]*\n(?:[ \t]*\n)*[ \t]*(\d+(?:\.\d+)?)(?:\s*/\s*10)?')
    last_match_val = None
    for match in pattern_next_line.finditer(clean_text):
        try:
            rating_val = float(match.group(1))
            if 1 <= rating_val <= 10:
                last_match_val = str(rating_val)
        except ValueError:
            continue
            
    if last_match_val:
        return last_match_val

    # Strategy 2: Number on the same line
    pattern_same_line = re.compile(r'\bRating\b\s*:?\s*(\d+(?:\.\d+)?)(?:\s*/\s*10)?')
    last_match_val = None
    for match in pattern_same_line.finditer(clean_text):
        try:
            rating_val = float(match.group(1))
            if 1 <= rating_val <= 10:
                last_match_val = str(rating_val)
        except ValueError:
            continue

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
                num_match = re.match(r'(\d+(?:\.\d+)?)(?:\s*/\s*10)?', next_line)
                if num_match:
                    try:
                        rating_val = float(num_match.group(1))
                        if 1 <= rating_val <= 10:
                            return str(rating_val)
                    except ValueError:
                        pass
                break
    
    # Fallback search in tail
    for num in ["10", "9", "8", "7", "6", "5", "4", "3", "2", "1"]:
        if num in tail:
            return num
    
    return "Error"

def extract_nested_json(text: str) -> Optional[Dict]:
    """
    Extract nested JSON from document extraction output.
    Based on retrieve_batch_results_extracted_document.ipynb
    """
    # Try to find a JSON object
    matches = re.finditer(r'(?m)^{.*}$', text, re.DOTALL)
    
    # Get the last match
    json_str = None
    for match in matches:
        json_str = match.group()
    
    if not json_str:
        # Try to find JSON between ```json and ```
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
    
    if not json_str:
        return None
        
    # Try to parse the extracted string as JSON
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Try to clean up the string
        try:
            json_str = json_str.strip()
            json_str = json_str.replace('\\n', '\n')
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None

def merge_nested_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """
    Merge two nested dictionaries recursively.
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
    """Process batch evaluations for document creation task."""

    def __init__(self, api_key: str = None):
        """Initialize the batch processor."""
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

        self.client = OpenAI(api_key=api_key)

    def check_existing_batch(self, batch_file: Path) -> Optional[Dict]:
        """
        Check if a batch has already been submitted for this file.

        Returns:
            Dict with batch info if exists, None otherwise
        """
        # Check for metadata file in standard location
        # Pattern: batch_prompts/{evaluation_type}/logs/{filename}.json
        metadata_path = batch_file.parent / "logs" / f"{batch_file.stem}.json"

        if metadata_path.exists():
            print(f"Found existing batch metadata: {metadata_path}")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            batch_id = metadata.get("batch_id")
            if batch_id:
                # Check batch status
                try:
                    batch = self.client.batches.retrieve(batch_id)
                    print(f"Existing batch found: {batch_id}")
                    print(f"Status: {batch.status}")

                    return {
                        "batch_id": batch_id,
                        "metadata_file": str(metadata_path),
                        "status": batch.status,
                        "batch": batch,
                        "metadata": metadata
                    }
                except Exception as e:
                    print(f"Could not retrieve batch {batch_id}: {e}")
                    print("Will create a new batch.")

        return None

    def submit_batch(self, batch_file: Path, description: str = None) -> Tuple[str, Dict]:
        """
        Submit a batch file for processing.
        
        Returns:
            Tuple of (batch_id, batch_info)
        """
        # Upload the batch file
        with open(batch_file, 'rb') as f:
            uploaded_file = self.client.files.create(
                file=f,
                purpose="batch"
            )
        
        # Create the batch
        batch = self.client.batches.create(
            input_file_id=uploaded_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": description or f"Document creation evaluation - {batch_file.name}",
                "source_file": str(batch_file)
            }
        )
        
        return batch.id, {
            "batch_id": batch.id,
            "input_file_id": uploaded_file.id,
            "status": batch.status,
            "created_at": batch.created_at
        }
    
    def check_batch_status(self, batch_id: str) -> Dict:
        """Check the status of a batch."""
        batch = self.client.batches.retrieve(batch_id)
        return {
            "status": batch.status,
            "completed_at": batch.completed_at,
            "failed_at": batch.failed_at,
            "expired_at": batch.expired_at,
            "request_counts": batch.request_counts.model_dump() if batch.request_counts else None,
            "output_file_id": batch.output_file_id,
            "error_file_id": batch.error_file_id
        }
    
    def retrieve_results(self, batch_id: str, evaluation_type: str) -> Dict:
        """
        Retrieve and process batch results based on evaluation type.

        Args:
            batch_id: The batch ID
            evaluation_type: One of 'document_rating', 'interaction_rating', 'extracted_document'
        """
        batch = self.client.batches.retrieve(batch_id)
        
        if batch.status != "completed":
            raise ValueError(f"Batch {batch_id} is not completed. Status: {batch.status}")
        
        if not batch.output_file_id:
            raise ValueError(f"No output file for batch {batch_id}")
        
        # Get the output file
        file_response = self.client.files.content(batch.output_file_id)
        
        results = {}
        errors = []
        
        # Process each line of results
        for line in file_response.text.strip().splitlines():
            try:
                result = json.loads(line)
                custom_id = result['custom_id']
                
                if result.get("error"):
                    errors.append({
                        "custom_id": custom_id,
                        "error": result["error"]
                    })
                    continue
                
                output = result["response"]["body"]["choices"][0]["message"]["content"]
                
                if not output:
                    errors.append({
                        "custom_id": custom_id,
                        "error": "Empty response"
                    })
                    continue
                
                # Process based on evaluation type
                if "rating" in evaluation_type:
                    # Extract rating for both document and interaction evaluations
                    rating = extract_rating(output)
                    
                    # Parse custom_id to get keys (format: model|doc_type|intent|worker_id)
                    parts = custom_id.split("|")
                    if len(parts) == 4:
                        model, doc_type, intent, worker_id = parts
                        
                        # Build nested structure
                        if model not in results:
                            results[model] = {}
                        if doc_type not in results[model]:
                            results[model][doc_type] = {}
                        if intent not in results[model][doc_type]:
                            results[model][doc_type][intent] = {}
                        
                        results[model][doc_type][intent][worker_id] = {
                            "extracted_rating": rating,
                            "output": output
                        }
                    else:
                        errors.append({
                            "custom_id": custom_id,
                            "error": f"Invalid custom_id format: {custom_id}"
                        })
                
                elif evaluation_type == "extracted_document":
                    # Extract document from JSON response
                    json_result = extract_nested_json(output)
                    
                    if json_result and "Final Document" in json_result:
                        final_document = json_result["Final Document"]
                        thought = json_result.get("Thought", "")
                    else:
                        # Fallback: try to extract document directly
                        parts = output.split('"Final Document": "')
                        if len(parts) > 1:
                            final_document = parts[1].split('"')[0]
                            thought = ""
                        else:
                            errors.append({
                                "custom_id": custom_id,
                                "error": "Could not extract document"
                            })
                            continue
                    
                    # Store with custom_id as key
                    results[custom_id] = {
                        "Final Document": final_document,
                        "Thought": thought,
                        "Output": output
                    }
                
            except Exception as e:
                errors.append({
                    "custom_id": custom_id if 'custom_id' in locals() else "unknown",
                    "error": str(e)
                })
        
        return {
            "results": results,
            "errors": errors,
            "total_processed": len(results),
            "total_errors": len(errors)
        }

def main():
    parser = argparse.ArgumentParser(
        description="Process batch evaluation for document creation."
    )
    
    parser.add_argument(
        "--batch_file",
        type=str,
        required=True,
        help="Path to the batch JSONL file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        help="Model to use for evaluation (only for submission)"
    )
    parser.add_argument(
        "--batch_id",
        type=str,
        help="Existing batch ID to check status or retrieve results"
    )
    parser.add_argument(
        "--log_file",
        type=str,
        help="Path to save batch log information"
    )
    parser.add_argument(
        "--check_status",
        action="store_true",
        help="Only check batch status without retrieving results"
    )
    parser.add_argument(
        "--poll_interval",
        type=int,
        default=30,
        help="Polling interval in seconds (default: 30)"
    )
    parser.add_argument(
        "--max_wait",
        type=int,
        default=7200,
        help="Maximum wait time in seconds (default: 7200 = 2 hours)"
    )
    parser.add_argument(
        "--no_check_existing",
        action="store_true",
        help="Don't check for existing batch, always create new batch"
    )

    args = parser.parse_args()
    
    # Initialize processor
    processor = BatchEvaluationProcessor()
    
    # Determine evaluation type from batch file path
    batch_file = Path(args.batch_file)
    # Check for directory names used by generate scripts (document_rating, interaction_rating, extracted_document)
    if "document_rating" in str(batch_file):
        evaluation_type = "document_rating"
    elif "interaction_rating" in str(batch_file):
        evaluation_type = "interaction_rating"
    elif "extracted_document" in str(batch_file) or "extract" in str(batch_file):
        evaluation_type = "extracted_document"
    else:
        # Fallback: infer from keywords
        if "document" in str(batch_file) and "rating" in str(batch_file):
            evaluation_type = "document_rating"
        elif "interaction" in str(batch_file) and "rating" in str(batch_file):
            evaluation_type = "interaction_rating"
        else:
            raise ValueError(f"Cannot determine evaluation type from batch file path: {batch_file}")
    
    print(f"Evaluation type: {evaluation_type}")

    # Check for existing batch before submitting
    existing_batch = None
    if not args.batch_id and not args.no_check_existing:
        existing_batch = processor.check_existing_batch(batch_file)

        if existing_batch:
            if existing_batch["status"] == "completed":
                print("Batch already completed! Retrieving results...")
                batch_id = existing_batch["batch_id"]
                # Skip to results retrieval (will happen after the polling section)
            elif existing_batch["status"] in ["failed", "expired", "cancelled"]:
                print(f"Previous batch {existing_batch['status']}. Creating new batch...")
                existing_batch = None  # Allow new submission
            elif existing_batch["status"] in ["validating", "in_progress", "finalizing"]:
                print(f"Batch is {existing_batch['status']}. Resuming wait for completion...")
                batch_id = existing_batch["batch_id"]
                # Will continue to polling section
            else:
                print(f"Unknown batch status: {existing_batch['status']}. Creating new batch...")
                existing_batch = None

    # Submit new batch or use existing
    if args.batch_id:
        batch_id = args.batch_id
        print(f"Using provided batch ID: {batch_id}")
    elif existing_batch:
        batch_id = existing_batch["batch_id"]
        print(f"Using existing batch: {batch_id}")
    else:
        print(f"Submitting batch file: {batch_file}")
        batch_id, batch_info = processor.submit_batch(
            batch_file,
            description=f"{evaluation_type} evaluation"
        )
        print(f"Batch submitted: {batch_id}")

        # Load keys file if it exists (for metadata)
        keys_file = batch_file.parent / f"{batch_file.stem}_keys.json"
        key_dict = {}
        if keys_file.exists():
            with open(keys_file, 'r') as f:
                keys_data = json.load(f)
                # Convert list keys to dict for document creation
                if isinstance(keys_data.get("keys"), list):
                    key_dict = {f"{k[0]}|{k[1]}|{k[2]}|{k[3]}": k
                               for k in keys_data["keys"]} if keys_data["keys"] and len(keys_data["keys"][0]) == 4 else {}

        # Prepare metadata
        metadata = {
            "batch_id": batch_id,
            "batch_info": batch_info,
            "evaluation_type": evaluation_type,
            "batch_file": str(batch_file),
            "submitted_at": datetime.now().isoformat(),
            "key_dict": key_dict
        }

        # Always save batch metadata to standard location
        metadata_path = batch_file.parent / "logs" / f"{batch_file.stem}.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Batch metadata saved to: {metadata_path}")

        # Also save to custom log file if requested (for backwards compatibility)
        if args.log_file:
            log_path = Path(args.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Log also saved to: {log_path}")
    
    # Check status only
    if args.check_status:
        status = processor.check_batch_status(batch_id)
        print(f"Batch status: {status['status']}")
        print(json.dumps(status, indent=2))
        return
    
    # Poll for completion
    print(f"Waiting for batch to complete...")
    start_time = time.time()
    
    while True:
        status = processor.check_batch_status(batch_id)
        
        if status["status"] == "completed":
            print("Batch completed!")
            break
        elif status["status"] in ["failed", "expired", "cancelled"]:
            print(f"Batch {status['status']}: {status}")
            sys.exit(1)
        
        elapsed = time.time() - start_time
        if elapsed > args.max_wait:
            print(f"Timeout waiting for batch completion ({args.max_wait} seconds)")
            sys.exit(1)
        
        print(f"Status: {status['status']} - waiting {args.poll_interval} seconds... (elapsed: {int(elapsed)}s)")
        time.sleep(args.poll_interval)
    
    # Retrieve and process results
    print("Retrieving results...")
    results_data = processor.retrieve_results(batch_id, evaluation_type)
    
    print(f"Processed: {results_data['total_processed']} results, {results_data['total_errors']} errors")
    
    if results_data['errors']:
        print("\nErrors encountered:")
        for error in results_data['errors'][:5]:  # Show first 5 errors
            print(f"  - {error['custom_id']}: {error['error']}")
        if len(results_data['errors']) > 5:
            print(f"  ... and {len(results_data['errors']) - 5} more errors")
    
    # Save results
    # Get base directory (evaluation/document_creation/)
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "evaluation_outputs" / evaluation_type
    output_dir.mkdir(parents=True, exist_ok=True)

    # Preserve directory structure (e.g., gpt-5-mini/simulation_name)
    # base_dir is evaluation/document_creation/, preserves gpt-5-mini/file structure
    # relative_to gives us: user_model/filename.jsonl from batch_prompts/{evaluation_type}/
    rel_path = batch_file.relative_to(batch_file.parent.parent)
    output_file = output_dir / rel_path.with_suffix('.json')

    # Ensure parent directory exists (handles subdirectories in file_name like "gpt-5-mini/...")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results if any
    existing_results = {}
    if output_file.exists():
        with open(output_file, 'r') as f:
            existing_data = json.load(f)
            # Handle wrapped format
            if "evaluations" in existing_data:
                existing_results = existing_data["evaluations"]
            elif "documents" in existing_data:
                existing_results = existing_data["documents"]
            else:
                existing_results = existing_data
    
    # Merge results
    if "rating" in evaluation_type:
        final_results = merge_nested_dicts(existing_results, results_data['results'])
        output_data = {
            "evaluations": final_results,
            "metadata": {
                "batch_id": batch_id,
                "evaluation_type": evaluation_type,
                "total_evaluations": results_data['total_processed'],
                "errors": results_data['total_errors'],
                "updated_at": datetime.now().isoformat()
            }
        }
    else:
        # For document extraction
        final_results = {**existing_results, **results_data['results']}
        output_data = {
            "documents": final_results,
            "metadata": {
                "batch_id": batch_id,
                "total_documents": len(final_results),
                "errors": results_data['total_errors'],
                "updated_at": datetime.now().isoformat()
            }
        }
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Update log file with completion info
    if args.log_file and Path(args.log_file).exists():
        with open(args.log_file, 'r') as f:
            log_data = json.load(f)
        
        log_data["completed_at"] = datetime.now().isoformat()
        log_data["output_file"] = str(output_file)
        log_data["results_summary"] = {
            "total_processed": results_data['total_processed'],
            "total_errors": results_data['total_errors']
        }
        
        with open(args.log_file, 'w') as f:
            json.dump(log_data, f, indent=2)

if __name__ == "__main__":
    main()