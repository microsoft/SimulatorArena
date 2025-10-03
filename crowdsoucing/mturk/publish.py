import argparse
import os
import boto3
from dotenv import load_dotenv

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["math", "doc"], help="Task type: 'math' or 'doc'")
    parser.add_argument("--mturk_region", default="us-east-1", help="MTurk region")
    parser.add_argument("--num_hits", type=int, default=1, help="Number of HITs")
    parser.add_argument("--num_assignments", type=int, default=1, help="Assignments per HIT")
    parser.add_argument("--live_mode", action="store_true", help="Run in live mode (charges account)")
    parser.add_argument("--no_qualification", action="store_true", help="Publish without qualifications (sandbox only)")
    parser.add_argument("--cheat", action="store_true", help="Enable cheat mode (sandbox only)")
    parser.add_argument("--provide_model", action="store_true", help="Add model info to URL")
    return parser.parse_args()

def get_mturk_client(args):
    env_path = os.path.expanduser('~/.env')
    load_dotenv(env_path)
    
    is_live = args.live_mode
    endpoint = f"https://mturk-requester{'' if is_live else '-sandbox'}.{args.mturk_region}.amazonaws.com"
    key = os.getenv("MTURK_KEY")
    secret = os.getenv("MTURK_SECRET")
    
    return boto3.client(
        "mturk",
        aws_access_key_id=key,
        aws_secret_access_key=secret,
        region_name=args.mturk_region,
        endpoint_url=endpoint
    )

def get_qualifications(live_mode, no_qualification):
    general_quals = [
        {
            "QualificationTypeId": "2F1QJWKUDD8XADTFD2Q0G6UTO95ALH",
            "Comparator": "Exists",
            'RequiredToPreview': True,
            'ActionsGuarded': 'DiscoverPreviewAndAccept'
        },
        {
            "QualificationTypeId": "00000000000000000071",
            "Comparator": "In",
            'LocaleValues': [{'Country': c} for c in ['US', 'GB', 'AU']],
            'RequiredToPreview': True,
            'ActionsGuarded': 'DiscoverPreviewAndAccept'
        },
        {
            'QualificationTypeId': '00000000000000000040',
            'Comparator': 'GreaterThanOrEqualTo',
            'IntegerValues': [1000],
            'RequiredToPreview': True,
            'ActionsGuarded': 'DiscoverPreviewAndAccept'
        },
        {
            'QualificationTypeId': '000000000000000000L0',
            'Comparator': 'GreaterThanOrEqualTo',
            'IntegerValues': [98],
            'RequiredToPreview': True,
            'ActionsGuarded': 'DiscoverPreviewAndAccept'
        }
    ]
    
    sandbox_qual = [] if no_qualification else [{
        "QualificationTypeId": "xxxxxxxxxxx",
        "Comparator": "EqualTo",
        "IntegerValues": [100],
        "RequiredToPreview": True,
        "ActionsGuarded": "DiscoverPreviewAndAccept",
    }]
    
    return general_quals if live_mode else sandbox_qual

def get_task_config(task_type):
    configs = {
        "math": {
            "base_url": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            "title": "Engaging Math problem solving with AI tutor (batch xx)",
            "description": "Pick math problems that you are interested in, and chat with AI tutor to learn to solve it. [Please view in light mode and Chrome]",
            "reward": "7.5",
            "keywords": "math, tutor, AI, problem solving, chat",
            "models": {
                "gpt-4o-mini": 1, "mistral-large-2407": 1, "claude-3-5-sonnet-20240620": 1,
                "llama-3-1-70b": 1, "llama-3-1-8b": 1, "phi-3-medium": 1,
                "phi-3-small": 1, "gpt-4o": 1, "gpt-4-turbo": 1
            },
            "cost_multiplier": 1.4,
            "cost_base": 6
        },
        "doc": {
            "base_url": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            "title": "Interactive document creation with AI writing assistant (batch xx)",
            "description": "This is the batch xx of the original document creation task. [Please view in light mode and Chrome]",
            "reward": "8",
            "keywords": "document creation, AI writing assistant, chat",
            "document_type": "creative_writing",
            "models": {
                "gpt-4o-mini": 1, "mistral-large-2407": 1, "claude-3-5-sonnet-20240620": 1,
                "llama-3-1-70b": 1, "llama-3-1-8b": 1, "phi-3-medium": 1,
                "phi-3-small": 1, "gpt-4o": 1, "gpt-4-turbo": 1
            },
            "cost_multiplier": 1.45,
            "cost_base": 6.5
        }
    }
    return configs[task_type]

def create_question(base_url, username, model=None, cheat=False, document_type=None):
    url = f"{base_url}?username={username}"
    if cheat:
        url += "&cheat=yes"
    if model:
        url += f"&model={model}"
    if document_type:
        url += f"&document_type={document_type}"
            
    return f"""
        <ExternalQuestion xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2006-07-14/ExternalQuestion.xsd">
            <ExternalURL>{url}</ExternalURL>
            <FrameHeight>1000</FrameHeight>
        </ExternalQuestion>
    """


def create_hits(mturk, config, args):
    username = "mturk" if args.live_mode else "mturk-sandbox"
    lifetime = 60 * 60 * (24 if args.live_mode else 24 * 4)
    duration = 60 * (60 if args.live_mode else 120)
    quals = get_qualifications(args.live_mode, args.no_qualification)
    
    if args.provide_model:
        hits = []
        for model, assignments in config["models"].items():
            hit = mturk.create_hit(
                Title=config["title"],
                Description=config["description"],
                Keywords=config["keywords"],
                Reward=config["reward"],
                MaxAssignments=assignments,
                LifetimeInSeconds=lifetime,
                AssignmentDurationInSeconds=duration,
                AutoApprovalDelayInSeconds=3600,
                Question=create_question(config["base_url"], username, model, cheat=args.cheat, document_type=config.get("document_type")),
                QualificationRequirements=quals
            )
            hits.append(hit)
        return hits
    else:
        last_hit = None
        for _ in range(args.num_hits):
            last_hit = mturk.create_hit(
                Title=config["title"],
                Description=config["description"],
                Keywords=config["keywords"],
                Reward=config["reward"],
                MaxAssignments=args.num_assignments,
                LifetimeInSeconds=lifetime,
                AssignmentDurationInSeconds=duration,
                AutoApprovalDelayInSeconds=3600,
                Question=create_question(config["base_url"], username, cheat=args.cheat, document_type=config.get("document_type")),
                QualificationRequirements=quals
            )
        return last_hit

def main():
    args = get_args()
    if args.cheat and args.live_mode:
        raise ValueError("Can't use --cheat in live mode")
    if args.no_qualification and args.live_mode:
        raise ValueError("Can't use --no_qualification in live mode")
    
    mturk = get_mturk_client(args)
    print(f"Account balance: ${mturk.get_account_balance()['AvailableBalance']}")
    
    config = get_task_config(args.task)
    total_assignments = sum(config["models"].values()) if args.provide_model else args.num_hits * args.num_assignments
    cost_estimate = config["cost_multiplier"] * config["cost_base"] * total_assignments
    
    if args.live_mode:
        print(f"Publishing {total_assignments} HITs. Estimated cost: ${cost_estimate:.2f}")
        if input("Continue? (yes/no): ") != "yes":
            return
            
    hit = create_hits(mturk, config, args)
    if hit:
        # If hit is a list, use the first hit for displaying the HIT Group Link.
        if isinstance(hit, list):
            hit = hit[0]
        env = "" if args.live_mode else "sandbox"
        print(f"HIT Group Link: https://worker{env}.mturk.com/mturk/preview?groupId={hit['HIT']['HITGroupId']}")
        print(f"Mode: {'cheat' if args.cheat else 'normal'}")


if __name__ == "__main__":
    main()
