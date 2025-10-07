import argparse
from os import path

import os

import boto3


from dotenv import load_dotenv
dotenv_path = os.path.expanduser('~/.env')
load_dotenv(dotenv_path)

parser = argparse.ArgumentParser()
parser.add_argument("--mturk_region", default="us-east-1", help="The region for mturk (default: us-east-1)")
parser.add_argument(
    "--live_mode",
    action="store_true",
    help="""
    Whether to run in live mode with real turkers. This will charge your account money.
    If you don't use this flag, the HITs will be deployed on the sandbox version of mturk,
    which will not charge your account money.
    """,
)

args = parser.parse_args()

MTURK_URL = f"https://mturk-requester{'' if args.live_mode else '-sandbox'}.{args.mturk_region}.amazonaws.com"
MTURK_KEY = os.getenv("MTURK_KEY")
MTURK_SECRET = os.getenv("MTURK_SECRET")

mturk = boto3.client(
    "mturk",
    aws_access_key_id=MTURK_KEY,
    aws_secret_access_key=MTURK_SECRET,
    region_name=args.mturk_region,
    endpoint_url=MTURK_URL,
)

print("I have $" + mturk.get_account_balance()['AvailableBalance'] + " in my account")

print(f"You are assigning qualification to the workers in the {'main' if args.live_mode else 'sandbox'} mode.")

# this is qualification type for turkers who done many hits
qualification_type_id = "xxxxxxxxxxxxxxx"

workers = [
    # "xxxxxxxxxxxxxxx",
]

for worker in workers:
    response = mturk.associate_qualification_with_worker(
        QualificationTypeId=qualification_type_id,
        WorkerId=worker,
        IntegerValue=100,
        SendNotification=False,
    )
    print(f"Assigned qualification to worker {worker}")