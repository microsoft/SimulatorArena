# Heroku Deployment Documentation

This directory contains multiple [Heroku](https://www.heroku.com/) applications for different crowdsourcing interfaces. Each interface follows the same deployment process.

## Prerequisites

1. Install the [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli)
2. Have a Heroku account
3. Python 3.12 installed locally
4. Git installed locally

## Project Structure

Each interface directory contains:
- `app.py` - Main Flask application
- `requirements.txt` - Python dependencies
- `Procfile` - Heroku process file
- `setup.sh` - Setup script for environment variables
- `.python-version` - Python version specification
- `utils.py` - Utility functions
- `logger_config.py` - Logging configuration
- `worker_user_id_dict.json` and `cookies.json` - Cookies that track user's finished hits
- Various data and resource directories (img/, data/)

## Deployment Steps

Follow these steps for each interface you want to deploy:

1. **Login to Heroku CLI**
   ```bash
   heroku login
   ```

2. **Create a new Heroku app**
   ```bash
   cd [interface_name]  # e.g., heroku_document_creation
   heroku create [app-name]    # Choose a unique app name
   ```

3. **Set up environment variables**
   ```bash
   # Review and modify setup.sh with your configuration
   chmod +x setup.sh
   ./setup.sh
   ```

4. **Deploy the application**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push heroku main
   ```

5. **Verify the deployment**
   ```bash
   heroku open
   ```

6. **Monitor the logs**
   ```bash
   heroku logs --tail
   ```

## Common Configuration

Each interface requires these common configuration steps:

1. **Environment Variables**
   - Check `setup.sh` for required environment variables
   - Set them in Heroku using the folloing code or set in the Heroku's web interface (these are safer than putting the variables in setup.sh):
     ```bash
     heroku config:set KEY=VALUE
     ```

2. **Python Dependencies**
   - All required packages are listed in `requirements.txt`
   - Heroku automatically installs them during deployment

3. **Procfile Configuration**
   - Each app uses a web dyno specified in `Procfile`
   - Format: `web: python app.py`

## Troubleshooting

Common issues and solutions:

1. **Application Error (H10)**
   - Check logs: `heroku logs --tail`
   - Verify Procfile configuration
   - Ensure all dependencies are in requirements.txt

2. **Build Failures**
   - Verify Python version in `.python-version`
   - Check for any syntax errors in app.py
   - Ensure all required files are committed

3. **Environment Variables**
   - Verify all required variables are set:
     ```bash
     heroku config
     ```

## Maintenance

- Monitor application performance using:
  ```bash
  heroku ps
  heroku logs --tail
  ```

- Update dependencies:
  ```bash
  pip freeze > requirements.txt
  git commit -am "Update dependencies"
  git push heroku main
  ```

## Local Development

To run any interface locally:

1. Create a new conda environment:
   ```bash
    conda create -n crowdsource-interface python=3.12

    # Activate the environment
    conda activate crowdsource-interface
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   gradio app.py
   ```