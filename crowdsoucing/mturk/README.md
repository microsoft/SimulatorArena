# Useful MTurk Scripts Documentation

## Prerequisites for All Scripts

AWS credentials in `~/.env`:
```
MTURK_KEY=your_aws_access_key
MTURK_SECRET=your_aws_secret_key
```

Required package: `boto3`, `python-dotenv`

## publish.py

Publishes HITs to Amazon Mechanical Turk with configurable task types and requirements.

### Usage
```bash
python publish.py --task TYPE [OPTIONS]

# Required:
--task          # Type of HIT (math or doc)

# Optional:
--mturk_region  # AWS region (default: us-east-1)
--num_hits      # Number of HITs (default: 1)
--live_mode     # Run in production mode
--provide_model # Add model information to URLs
```

Example:
```bash
python publish.py --task math --live_mode --num_hits 5
```

## publish_bonus_hit.py

Creates compensation HITs for specific workers.

### Usage
```bash
python publish_bonus_hit.py [OPTIONS]

# Optional:
--mturk_region        # AWS region (default: us-east-1)
--num_hits           # Number of HITs (default: 1)
--live_mode          # Run in production mode
--qualification_name # Name for qualification
```

Configuration:
- Edit `worker_ids` list in script
- Default reward: $6.50
- HIT lifetime: 7 days
- Assignment duration: 10 minutes

## qualification.py

Assigns qualifications to specific MTurk workers.

### Usage
```bash
python qualification.py [OPTIONS]

# Optional:
--mturk_region  # AWS region (default: us-east-1)
--live_mode     # Run in production mode
```

Configuration:
- Edit `workers` list in script
- Set `qualification_type_id` 
- Default qualification value: 100