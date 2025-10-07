#!/usr/bin/env python3

import argparse
import os
import boto3
from dotenv import load_dotenv

# -----------------------
# 1. Load environment
# -----------------------
dotenv_path = os.path.expanduser("~/.env")
load_dotenv(dotenv_path)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mturk_region",
    default="us-east-1",
    help="The region for mturk (default: us-east-1)",
)
parser.add_argument(
    "--num_hits",
    type=int,
    default=1,
    help="The number of HITs to publish (default: 1)."
)
parser.add_argument(
    "--live_mode",
    action="store_true",
    help="""
    Whether to run in live mode with real Turkers. This will charge your account money.
    If not used, the HITs will be deployed on the sandbox version of MTurk,
    which does not charge your account money.
    """,
)
parser.add_argument(
    "--qualification_name",
    default="CompensationQualification",
    help="A name for the newly created qualification type.",
)
args = parser.parse_args()

# -----------------------
# 2. Set endpoint & keys
# -----------------------
MTURK_URL = (
    f"https://mturk-requester.{args.mturk_region}.amazonaws.com"
    if args.live_mode
    else f"https://mturk-requester-sandbox.{args.mturk_region}.amazonaws.com"
)

MTURK_KEY = os.getenv("MTURK_KEY")
MTURK_SECRET = os.getenv("MTURK_SECRET")

mturk = boto3.client(
    "mturk",
    aws_access_key_id=MTURK_KEY,
    aws_secret_access_key=MTURK_SECRET,
    region_name=args.mturk_region,
    endpoint_url=MTURK_URL,
)

# -----------------------
# 3. Worker IDs to reward
# -----------------------
# Adjust this list to include the Worker IDs you want to compensate
worker_ids = [
    ""
]
num_workers = len(worker_ids)

# -----------------------
# 4. Print account balance
# -----------------------
balance = mturk.get_account_balance()["AvailableBalance"]
print(f"I have ${balance} in my account.")

# -----------------------
# 5. Estimate cost
# -----------------------
# Adjust the reward amount here if needed.
# Example: $6.5 per HIT * number of hits * number of assignments
reward_amount = 6.5
cost_estimate = reward_amount * args.num_hits * num_workers
if args.live_mode:
    print(
        f"You are about to publish {args.num_hits} HIT(s) with {num_workers} assignment(s) each, "
        f"total {args.num_hits * num_workers} assignment(s).\n"
        f"This will charge approximately ${cost_estimate} from your account."
    )
    response = input("Are you sure you want to continue? (yes/no): ")
    if response.lower() != "yes":
        print("Exiting without publishing HITs.")
        exit()

# -----------------------
# 6. Create a new qualification
# -----------------------
# This will create a new qualification that we can assign to the worker IDs.
# If you need to create only once (instead of every time), handle the logic 
# to check if it already exists. For simplicity, we'll just create new each time.
qual_response = mturk.create_qualification_type(
    Name=args.qualification_name,
    Description="Qualification used to control access to compensation HIT.",
    QualificationTypeStatus="Active",
    Keywords="compensation, bonus"
)
qualification_id = qual_response["QualificationType"]["QualificationTypeId"]
print(f"Created new qualification: {args.qualification_name} (ID: {qualification_id})")

# -----------------------
# 7. Associate qualification to each worker
# -----------------------
for w in worker_ids:
    try:
        mturk.associate_qualification_with_worker(
            QualificationTypeId=qualification_id,
            WorkerId=w,
            IntegerValue=100,  # or any integer that you want
            SendNotification=False
        )
        print(f"Assigned qualification to worker {w}.")
    except Exception as e:
        print(f"Error assigning qualification to worker {w}: {str(e)}")

# -----------------------
# 8. Define the question XML
# -----------------------
# Minimal QuestionForm XML for a "no-task" HIT
question_xml = """<QuestionForm 
    xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2017-11-06/QuestionForm.xsd">
  <Question>
    <QuestionIdentifier>compensation_question</QuestionIdentifier>
    <DisplayName>Compensation Question</DisplayName>
    <IsRequired>true</IsRequired>
    <QuestionContent>
      <Text>This is a compensation HIT. Simply type anything and click "Submit" to finish.</Text>
    </QuestionContent>
    <AnswerSpecification>
      <FreeTextAnswer/>
    </AnswerSpecification>
  </Question>
</QuestionForm>"""

# -----------------------
# 9. Qualification requirement
# -----------------------
qualification_requirements = [
    {
        "QualificationTypeId": qualification_id,
        "Comparator": "EqualTo",
        "IntegerValues": [100],
        "RequiredToPreview": True,
        "ActionsGuarded": "DiscoverPreviewAndAccept",
    }
]

# -----------------------
# 10. Create the HIT(s)
# -----------------------
for i in range(args.num_hits):
    new_hit = mturk.create_hit(
        Title="Compensation HIT - No task, just submit",
        Description="This HIT is only for compensating users who experienced an issue in a previous task.",
        Keywords="compensation, bonus, no-task",
        Reward=str(reward_amount),  # Convert float to string
        MaxAssignments=num_workers,
        LifetimeInSeconds=60 * 60 * 24 * 7,  # 7 days to accept the HIT
        AssignmentDurationInSeconds=60 * 10,  # 10 minutes once accepted
        AutoApprovalDelayInSeconds=60 * 60,  # Auto-approve in 1 hour
        Question=question_xml,
        QualificationRequirements=qualification_requirements,
    )

    hit_id = new_hit["HIT"]["HITId"]
    group_id = new_hit["HIT"]["HITGroupId"]
    if args.live_mode:
        preview_url = f"https://worker.mturk.com/mturk/preview?groupId={group_id}"
    else:
        preview_url = f"https://workersandbox.mturk.com/mturk/preview?groupId={group_id}"

    print(f"\nCreated compensation HIT {i+1}/{args.num_hits}:")
    print(f"  HITId: {hit_id}")
    print(f"  Preview URL: {preview_url}")

print("\nDone. All compensation HIT(s) created.")