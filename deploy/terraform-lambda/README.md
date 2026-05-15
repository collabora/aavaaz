# Deploying Aavaaz to AWS Lambda

This guide deploys the Aavaaz transcription web demo on AWS Lambda (CPU, free tier compatible).

## Prerequisites

### 1. Install AWS CLI v2

```bash
# Linux (x86_64)
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Verify
aws --version
```

For other platforms, see: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html

### 2. Install Terraform

```bash
# Fedora
sudo dnf install -y dnf-plugins-core
sudo dnf config-manager addrepo --from-repofile=https://rpm.releases.hashicorp.com/fedora/hashicorp.repo
sudo dnf install -y terraform

# Ubuntu/Debian
wget -O - https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
sudo apt update && sudo apt install terraform
```

### 3. Install Docker

Ensure Docker is installed and running: https://docs.docker.com/engine/install/

### 4. Configure AWS Credentials

#### Create an IAM user (recommended over using root)

1. Log in to the [AWS Console](https://console.aws.amazon.com/) as root
2. Go to **IAM** → **Users** → **Create user**
3. Name: `aavaaz-deploy`
4. Click **Next** → select **Attach policies directly**
5. Attach these managed policies:
   - `AmazonEC2ContainerRegistryFullAccess`
   - `AWSLambda_FullAccess`
   - `AmazonS3FullAccess`
   - `AmazonAPIGatewayAdministrator`
   - `IAMFullAccess`
6. Click **Next** → **Create user**
7. Click the new user → **Security credentials** tab
8. Under **Access keys**, click **Create access key**
9. Select **Command Line Interface (CLI)** → acknowledge → **Create**
10. **Copy both the Access Key ID and Secret Access Key** (secret is shown only once!)

#### Configure the CLI

```bash
aws configure
```

When prompted:
- **AWS Access Key ID:** paste your key
- **AWS Secret Access Key:** paste your secret
- **Default region name:** `us-east-1`
- **Default output format:** `json`

Verify:
```bash
aws sts get-caller-identity
```

## Deploy

```bash
cd deploy/terraform-lambda
./deploy.sh
```

This script:
1. Runs `terraform apply` to create infrastructure (ECR, Lambda, S3, API Gateway, IAM roles)
2. Copies WhisperLive source from `~/src/WhisperLive/whisper_live/` into the build context
3. Builds the Docker container image with the Whisper model baked in
4. Pushes the image to ECR
5. Updates the Lambda function
6. Prints the web demo URL and API endpoint

## Important: Cold Start

The first request after the Lambda has been idle (~5-15 minutes) will **timeout** through API Gateway (hard 29-second limit). This is because loading the Whisper model on CPU takes ~20-40 seconds.

**Workaround — warm up the Lambda before using:**
```bash
aws lambda invoke --function-name aavaaz-transcribe \
  --payload '{}' --cli-binary-format raw-in-base64-out /tmp/warmup.json
```

After warming up, responses are fast (1-5 seconds). The Lambda stays warm for 5-15 minutes between requests.

## Usage

After deployment, visit the **web demo URL** printed at the end. You'll see a drag-and-drop interface where you can upload audio files for transcription.

API endpoint (OpenAI-compatible):
```bash
curl -X POST "$API_ENDPOINT" \
  -F "file=@audio.wav" \
  -F "response_format=text"
```

## Free Tier Limits

| Resource | Free Tier Limit | Notes |
|----------|----------------|-------|
| Lambda requests | 1M/month | Each transcription = 1 request |
| Lambda compute | 400,000 GB-sec/month | At 3GB RAM: ~133K seconds (~37 hrs) |
| API Gateway | 1M calls/month | GET + POST combined |
| S3 | 5GB storage | Audio + transcript storage |
| ECR | 500MB storage | Container image |

With `tiny.en` at 3GB RAM on CPU, transcription runs at roughly 6x realtime (10s audio ≈ 1.5s processing), so free tier gives you approximately **37 hours of Lambda compute per month**.

## Configuration

Edit `deploy/terraform-lambda/main.tf` variables or set via Terraform:

```bash
terraform apply -var="whisper_model=base.en" -var="lambda_memory_mb=2048"
```

| Variable | Default | Description |
|----------|---------|-------------|
| `whisper_model` | `small.en` | Whisper model (tiny.en, base.en, small.en) |
| `lambda_memory_mb` | 3008 | Lambda memory (more = more CPU, max 3008 free tier) |
| `lambda_timeout` | 300 | Max seconds per invocation |
| `output_format` | json | Transcript format |
| `enable_api_gateway` | true | Create HTTP API |

**Note:** The deploy script uses `WHISPER_MODEL` env var to control the Docker build model. The Terraform variable controls the Lambda environment variable (they should match).

## Architecture

```
Browser → API Gateway (HTTPS) → Lambda (container)
                                    ├── WhisperLive transcriber (from ~/src/WhisperLive)
                                    ├── faster-whisper + CTranslate2 (CPU, int8)
                                    └── Aavaaz post-processing (formatting, PII redaction)
```

The Lambda container includes:
- Aavaaz package (post-processing, formatters)
- WhisperLive source (transcription engine) — copied during Docker build
- Pre-downloaded Whisper model cached at `/opt/hf_cache/`
- ffmpeg for audio format conversion

## Troubleshooting

### "Service Unavailable" on first request
Cold start exceeded API Gateway's 29s timeout. Warm up the Lambda first (see above).

### "Internal Server Error"
Check CloudWatch logs:
```bash
aws logs tail /aws/lambda/aavaaz-transcribe --since 5m
```

### Function URL returns 403
Some AWS accounts restrict public Lambda Function URLs. Use the API Gateway endpoint instead.

## Cleanup

```bash
cd deploy/terraform-lambda
terraform destroy
```

This removes all AWS resources (Lambda, ECR, S3 buckets, API Gateway, IAM roles).
