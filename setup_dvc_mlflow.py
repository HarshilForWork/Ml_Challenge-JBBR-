"""
Setup script for DVC and MLflow configuration
Run this after installing requirements.txt
"""
import os
import subprocess
import json
from pathlib import Path


def load_aws_credentials():
    """Load AWS credentials from aws_iam_key.json"""
    try:
        with open('aws_iam_key.json', 'r') as f:
            creds = json.load(f)
        return creds.get('access_key'), creds.get('Secret_access_key')
    except Exception as e:
        print(f"Error loading AWS credentials: {e}")
        return None, None


def setup_environment():
    """Set up environment variables for DVC and MLflow"""
    print("=" * 60)
    print("Setting up environment variables")
    print("=" * 60)
    
    # Load AWS credentials
    access_key, secret_key = load_aws_credentials()
    
    if not access_key or not secret_key:
        print("❌ Could not load AWS credentials from aws_iam_key.json")
        return False
    
    # Create .env file
    env_content = f"""# AWS Credentials for DVC
AWS_ACCESS_KEY_ID={access_key}
AWS_SECRET_ACCESS_KEY={secret_key}
AWS_DEFAULT_REGION=us-east-1

# MLflow Tracking (DagsHub)
MLFLOW_TRACKING_URI=https://dagshub.com/your-username/Ml_Challenge-JBBR-.mlflow
MLFLOW_TRACKING_USERNAME=your_dagshub_username
MLFLOW_TRACKING_PASSWORD=your_dagshub_token
MLFLOW_EXPERIMENT_NAME=prompt-quality-prediction
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("✓ Environment file created: .env")
    print("\n⚠️  IMPORTANT: Set these environment variables in PowerShell:")
    print(f'   $env:AWS_ACCESS_KEY_ID="{access_key}"')
    print(f'   $env:AWS_SECRET_ACCESS_KEY="{secret_key}"')
    print("\nOr activate .env before running dvc commands")
    
    return True


def test_dvc_connection():
    """Test DVC connection to S3"""
    print("\n" + "=" * 60)
    print("Testing DVC connection")
    print("=" * 60)
    
    # Load and set credentials
    access_key, secret_key = load_aws_credentials()
    if access_key and secret_key:
        os.environ['AWS_ACCESS_KEY_ID'] = access_key
        os.environ['AWS_SECRET_ACCESS_KEY'] = secret_key
    
    try:
        result = subprocess.run(['dvc', 'status', '-r', 's3storage'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ DVC connection successful!")
            return True
        else:
            print(f"⚠️  DVC status check returned: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error testing DVC: {e}")
        return False


def setup_dvc_credentials():
    """Configure DVC with AWS credentials locally"""
    print("\n" + "=" * 60)
    print("Configuring DVC with AWS credentials")
    print("=" * 60)
    
    access_key, secret_key = load_aws_credentials()
    if not access_key or not secret_key:
        print("❌ Could not load AWS credentials")
        return False
    
    try:
        # Store credentials in local config (not tracked by git)
        subprocess.run([
            'dvc', 'remote', 'modify', '--local', 's3storage', 
            'access_key_id', access_key
        ], check=True)
        
        subprocess.run([
            'dvc', 'remote', 'modify', '--local', 's3storage', 
            'secret_access_key', secret_key
        ], check=True)
        
        print("✓ DVC credentials configured locally")
        return True
    except Exception as e:
        print(f"❌ Error configuring DVC: {e}")
        return False


def setup_mlflow():
    """Configure MLflow for DagsHub"""
    print("\n" + "=" * 60)
    print("MLflow Configuration")
    print("=" * 60)
    
    print("\nTo use MLflow with DagsHub:")
    print("1. Sign up at https://dagshub.com")
    print("2. Connect your GitHub account")
    print("3. Import this repository: HarshilForWork/Ml_Challenge-JBBR-")
    print("4. Get your credentials from Settings > Integrations > MLflow")
    print("\n5. Update .env file with your DagsHub username and token")
    print("\nMLflow will track all your experiments automatically when you run training!")


def main():
    """Main setup function"""
    print("\n🚀 ML Challenge - Complete Setup")
    print("=" * 60)
    
    # 1. Setup environment file
    print("\n[1/3] Setting up environment file...")
    if not setup_environment():
        print("⚠️  Warning: Environment setup incomplete")
    
    # 2. Configure DVC
    print("\n[2/3] Configuring DVC...")
    if setup_dvc_credentials():
        test_dvc_connection()
    
    # 3. Setup MLflow
    print("\n[3/3] MLflow setup...")
    setup_mlflow()
    
    print("\n" + "=" * 60)
    print("✅ Setup Complete!")
    print("=" * 60)
    
    print("\n📊 What's configured:")
    print("  ✓ DVC - Data version control with S3")
    print("  ✓ MLflow - Experiment tracking (needs DagsHub setup)")
    print("  ✓ AWS credentials - Stored locally")
    
    print("\n🚀 Ready to use:")
    print("  dvc pull       # Download data from S3")
    print("  dvc push       # Upload data to S3")
    print("  python src/... # Run your scripts")


if __name__ == "__main__":
    main()
