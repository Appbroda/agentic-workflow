#!/usr/bin/env bash
# One-time AWS provisioning runbook for the agent.appbroda.com deployment.
# Idempotent: re-running skips anything that already exists rather than failing or duplicating.
#
# Requires: aws CLI configured with sufficient privileges, run from the repository root (or
# anywhere — it does not depend on the working directory).
set -euo pipefail

# ---------------------------------------------------------------------------
# Fixed parameters for this deployment
# ---------------------------------------------------------------------------
REGION="ap-south-1"
ACCOUNT_ID="118355822219"
VPC_ID="vpc-0b591fe66d9c4f3f6"
PUBLIC_SUBNET_ID="subnet-0040ac15b9dbb59c7"   # ap-south-1c, public
HOSTED_ZONE_ID="Z059084821WIGVKCZFTEK"        # appbroda.com
DOMAIN="agent.appbroda.com"
AURORA_CLUSTER_ID="ai-platform"
AURORA_DB_INSTANCE_ID="database-1-instance-1"
AURORA_DB_USER="platform"
ECR_REPO_NAME="ai-platform"
INSTANCE_NAME="ai-platform-prod"
INSTANCE_TYPE="t3.large"
GITHUB_ORG_REPO="Appbroda/agentic-workflow"
SG_NAME="ai-platform-sg"
EC2_ROLE_NAME="ai-platform-ec2-role"
GHA_ROLE_NAME="github-actions-deploy-ai-platform"

echo "== Region $REGION, account $ACCOUNT_ID =="

# ---------------------------------------------------------------------------
# ECR repository
# ---------------------------------------------------------------------------
if ! aws ecr describe-repositories --region "$REGION" --repository-names "$ECR_REPO_NAME" >/dev/null 2>&1; then
  echo "Creating ECR repo $ECR_REPO_NAME"
  aws ecr create-repository --region "$REGION" --repository-name "$ECR_REPO_NAME" \
    --image-scanning-configuration scanOnPush=true >/dev/null
else
  echo "ECR repo $ECR_REPO_NAME already exists"
fi
ECR_REPO_ARN="arn:aws:ecr:${REGION}:${ACCOUNT_ID}:repository/${ECR_REPO_NAME}"

# ---------------------------------------------------------------------------
# Security group: 80/443 from anywhere, no port 22 (SSM only)
# ---------------------------------------------------------------------------
SG_ID=$(aws ec2 describe-security-groups --region "$REGION" \
  --filters "Name=vpc-id,Values=$VPC_ID" "Name=group-name,Values=$SG_NAME" \
  --query "SecurityGroups[0].GroupId" --output text 2>/dev/null || echo "None")

if [ "$SG_ID" = "None" ] || [ -z "$SG_ID" ]; then
  echo "Creating security group $SG_NAME"
  SG_ID=$(aws ec2 create-security-group --region "$REGION" \
    --group-name "$SG_NAME" --description "agent.appbroda.com: 80/443 only, no SSH" \
    --vpc-id "$VPC_ID" --query "GroupId" --output text)
  aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SG_ID" \
    --ip-permissions \
      'IpProtocol=tcp,FromPort=80,ToPort=80,IpRanges=[{CidrIp=0.0.0.0/0,Description="http, redirects to https"}]' \
      'IpProtocol=tcp,FromPort=443,ToPort=443,IpRanges=[{CidrIp=0.0.0.0/0,Description="https"}]' >/dev/null
else
  echo "Security group $SG_NAME already exists ($SG_ID)"
fi

# ---------------------------------------------------------------------------
# Aurora: resource id for the IAM-auth policy, and the MinCapacity fix
# ---------------------------------------------------------------------------
DBI_RESOURCE_ID=$(aws rds describe-db-instances --region "$REGION" \
  --db-instance-identifier "$AURORA_DB_INSTANCE_ID" \
  --query "DBInstances[0].DbiResourceId" --output text)
echo "Aurora DbiResourceId: $DBI_RESOURCE_ID"

echo "Raising Aurora MinCapacity to 0.5 ACU (removes the 5-minute auto-pause cold start)"
aws rds modify-db-cluster --region "$REGION" \
  --db-cluster-identifier "$AURORA_CLUSTER_ID" \
  --serverless-v2-scaling-configuration MinCapacity=0.5,MaxCapacity=16 \
  --apply-immediately >/dev/null

# ---------------------------------------------------------------------------
# EC2 instance role + instance profile
# ---------------------------------------------------------------------------
if ! aws iam get-role --role-name "$EC2_ROLE_NAME" >/dev/null 2>&1; then
  echo "Creating IAM role $EC2_ROLE_NAME"
  aws iam create-role --role-name "$EC2_ROLE_NAME" \
    --assume-role-policy-document '{
      "Version": "2012-10-17",
      "Statement": [{"Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"}, "Action": "sts:AssumeRole"}]
    }' >/dev/null
else
  echo "IAM role $EC2_ROLE_NAME already exists"
fi

aws iam attach-role-policy --role-name "$EC2_ROLE_NAME" \
  --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore

aws iam put-role-policy --role-name "$EC2_ROLE_NAME" --policy-name "ai-platform-app-access" \
  --policy-document "{
    \"Version\": \"2012-10-17\",
    \"Statement\": [
      {
        \"Sid\": \"RdsIamAuth\",
        \"Effect\": \"Allow\",
        \"Action\": \"rds-db:connect\",
        \"Resource\": \"arn:aws:rds-db:${REGION}:${ACCOUNT_ID}:dbi-resource-id/${DBI_RESOURCE_ID}/${AURORA_DB_USER}\"
      },
      {
        \"Sid\": \"AppSecrets\",
        \"Effect\": \"Allow\",
        \"Action\": \"secretsmanager:GetSecretValue\",
        \"Resource\": \"arn:aws:secretsmanager:${REGION}:${ACCOUNT_ID}:secret:prod/ai-platform/*\"
      },
      {
        \"Sid\": \"EcrAuth\",
        \"Effect\": \"Allow\",
        \"Action\": \"ecr:GetAuthorizationToken\",
        \"Resource\": \"*\"
      },
      {
        \"Sid\": \"EcrPull\",
        \"Effect\": \"Allow\",
        \"Action\": [\"ecr:BatchGetImage\", \"ecr:GetDownloadUrlForLayer\", \"ecr:BatchCheckLayerAvailability\"],
        \"Resource\": \"${ECR_REPO_ARN}\"
      }
    ]
  }" >/dev/null

if ! aws iam get-instance-profile --instance-profile-name "$EC2_ROLE_NAME" >/dev/null 2>&1; then
  echo "Creating instance profile $EC2_ROLE_NAME"
  aws iam create-instance-profile --instance-profile-name "$EC2_ROLE_NAME" >/dev/null
  aws iam add-role-to-instance-profile --instance-profile-name "$EC2_ROLE_NAME" --role-name "$EC2_ROLE_NAME"
  echo "Waiting for instance profile propagation..."
  sleep 15
else
  echo "Instance profile $EC2_ROLE_NAME already exists"
fi

# ---------------------------------------------------------------------------
# Secrets Manager entries
# ---------------------------------------------------------------------------
if ! aws secretsmanager describe-secret --region "$REGION" --secret-id "prod/ai-platform/secret_encryption_key" >/dev/null 2>&1; then
  KEY_VALUE="v1:$(openssl rand -base64 32)"
  echo "Creating prod/ai-platform/secret_encryption_key"
  aws secretsmanager create-secret --region "$REGION" \
    --name "prod/ai-platform/secret_encryption_key" \
    --secret-string "$KEY_VALUE" >/dev/null
else
  echo "prod/ai-platform/secret_encryption_key already exists"
fi

if ! aws secretsmanager describe-secret --region "$REGION" --secret-id "prod/ai-platform/bootstrap_admin_password" >/dev/null 2>&1; then
  BOOTSTRAP_PW=$(openssl rand -base64 24)
  echo "Creating prod/ai-platform/bootstrap_admin_password"
  aws secretsmanager create-secret --region "$REGION" \
    --name "prod/ai-platform/bootstrap_admin_password" \
    --secret-string "$BOOTSTRAP_PW" >/dev/null
else
  echo "prod/ai-platform/bootstrap_admin_password already exists"
fi

# ---------------------------------------------------------------------------
# GitHub Actions OIDC provider + deploy role (no stored AWS keys in the repo)
# ---------------------------------------------------------------------------
OIDC_ARN="arn:aws:iam::${ACCOUNT_ID}:oidc-provider/token.actions.githubusercontent.com"
if ! aws iam get-open-id-connect-provider --open-id-connect-provider-arn "$OIDC_ARN" >/dev/null 2>&1; then
  echo "Creating GitHub Actions OIDC provider"
  aws iam create-open-id-connect-provider \
    --url "https://token.actions.githubusercontent.com" \
    --client-id-list "sts.amazonaws.com" \
    --thumbprint-list "6938fd4d98bab03faadb97b34396831e3780aea1" >/dev/null
else
  echo "GitHub Actions OIDC provider already exists"
fi

if ! aws iam get-role --role-name "$GHA_ROLE_NAME" >/dev/null 2>&1; then
  echo "Creating IAM role $GHA_ROLE_NAME"
  aws iam create-role --role-name "$GHA_ROLE_NAME" \
    --assume-role-policy-document "{
      \"Version\": \"2012-10-17\",
      \"Statement\": [{
        \"Effect\": \"Allow\",
        \"Principal\": {\"Federated\": \"${OIDC_ARN}\"},
        \"Action\": \"sts:AssumeRoleWithWebIdentity\",
        \"Condition\": {
          \"StringEquals\": {\"token.actions.githubusercontent.com:aud\": \"sts.amazonaws.com\"},
          \"StringLike\": {\"token.actions.githubusercontent.com:sub\": \"repo:${GITHUB_ORG_REPO}:ref:refs/heads/master\"}
        }
      }]
    }" >/dev/null
else
  echo "IAM role $GHA_ROLE_NAME already exists"
fi

aws iam put-role-policy --role-name "$GHA_ROLE_NAME" --policy-name "deploy-ai-platform" \
  --policy-document "{
    \"Version\": \"2012-10-17\",
    \"Statement\": [
      {\"Sid\": \"EcrAuth\", \"Effect\": \"Allow\", \"Action\": \"ecr:GetAuthorizationToken\", \"Resource\": \"*\"},
      {
        \"Sid\": \"EcrPush\",
        \"Effect\": \"Allow\",
        \"Action\": [
          \"ecr:BatchGetImage\", \"ecr:GetDownloadUrlForLayer\", \"ecr:BatchCheckLayerAvailability\",
          \"ecr:PutImage\", \"ecr:InitiateLayerUpload\", \"ecr:UploadLayerPart\", \"ecr:CompleteLayerUpload\"
        ],
        \"Resource\": \"${ECR_REPO_ARN}\"
      },
      {
        \"Sid\": \"FindInstance\",
        \"Effect\": \"Allow\",
        \"Action\": \"ec2:DescribeInstances\",
        \"Resource\": \"*\"
      },
      {
        \"Sid\": \"Redeploy\",
        \"Effect\": \"Allow\",
        \"Action\": [\"ssm:SendCommand\", \"ssm:GetCommandInvocation\"],
        \"Resource\": \"*\"
      }
    ]
  }" >/dev/null

# ---------------------------------------------------------------------------
# EC2 instance (only if not already running)
# ---------------------------------------------------------------------------
EXISTING_INSTANCE=$(aws ec2 describe-instances --region "$REGION" \
  --filters "Name=tag:Name,Values=$INSTANCE_NAME" "Name=instance-state-name,Values=pending,running,stopping,stopped" \
  --query "Reservations[0].Instances[0].InstanceId" --output text 2>/dev/null || echo "None")

if [ "$EXISTING_INSTANCE" != "None" ] && [ -n "$EXISTING_INSTANCE" ]; then
  echo "Instance $INSTANCE_NAME already exists ($EXISTING_INSTANCE) — skipping launch"
  INSTANCE_ID="$EXISTING_INSTANCE"
else
  AMI_ID=$(aws ssm get-parameter --region "$REGION" \
    --name "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64" \
    --query "Parameter.Value" --output text)
  echo "Launching $INSTANCE_TYPE ($AMI_ID) as $INSTANCE_NAME"

  USER_DATA=$(cat <<'USERDATA'
#!/bin/bash
set -euo pipefail
dnf install -y docker
systemctl enable --now docker
curl -SL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 \
  -o /usr/libexec/docker/cli-plugins/docker-compose
chmod +x /usr/libexec/docker/cli-plugins/docker-compose

# Format (only if unformatted) and mount the workspace volume.
DEVICE=$(readlink -f /dev/xvdf 2>/dev/null || readlink -f /dev/nvme1n1 2>/dev/null || echo /dev/xvdf)
if ! blkid "$DEVICE" >/dev/null 2>&1; then
  mkfs -t ext4 "$DEVICE"
fi
mkdir -p /mnt/workspaces
mount "$DEVICE" /mnt/workspaces
echo "$DEVICE /mnt/workspaces ext4 defaults,nofail 0 2" >> /etc/fstab

mkdir -p /opt/agentic-workflow
USERDATA
)

  INSTANCE_ID=$(aws ec2 run-instances --region "$REGION" \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --subnet-id "$PUBLIC_SUBNET_ID" \
    --associate-public-ip-address \
    --security-group-ids "$SG_ID" \
    --iam-instance-profile "Name=${EC2_ROLE_NAME}" \
    --block-device-mappings \
      'DeviceName=/dev/xvda,Ebs={VolumeSize=30,VolumeType=gp3,Encrypted=true}' \
      'DeviceName=/dev/xvdf,Ebs={VolumeSize=50,VolumeType=gp3,Encrypted=true}' \
    --user-data "$USER_DATA" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${INSTANCE_NAME}}]" \
    --query "Instances[0].InstanceId" --output text)
  echo "Launched $INSTANCE_ID — waiting for it to enter 'running' state"
  aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE_ID"
fi

# ---------------------------------------------------------------------------
# Elastic IP
# ---------------------------------------------------------------------------
EIP_ALLOC=$(aws ec2 describe-addresses --region "$REGION" \
  --filters "Name=tag:Name,Values=$INSTANCE_NAME" \
  --query "Addresses[0].AllocationId" --output text 2>/dev/null || echo "None")

if [ "$EIP_ALLOC" = "None" ] || [ -z "$EIP_ALLOC" ]; then
  echo "Allocating Elastic IP"
  EIP_ALLOC=$(aws ec2 allocate-address --region "$REGION" --domain vpc \
    --tag-specifications "ResourceType=elastic-ip,Tags=[{Key=Name,Value=${INSTANCE_NAME}}]" \
    --query "AllocationId" --output text)
fi

aws ec2 associate-address --region "$REGION" --instance-id "$INSTANCE_ID" --allocation-id "$EIP_ALLOC" >/dev/null
ELASTIC_IP=$(aws ec2 describe-addresses --region "$REGION" --allocation-ids "$EIP_ALLOC" \
  --query "Addresses[0].PublicIp" --output text)
echo "Elastic IP: $ELASTIC_IP"

# ---------------------------------------------------------------------------
# Route 53 A record
# ---------------------------------------------------------------------------
echo "Upserting $DOMAIN -> $ELASTIC_IP"
aws route53 change-resource-record-sets --hosted-zone-id "$HOSTED_ZONE_ID" \
  --change-batch "{
    \"Changes\": [{
      \"Action\": \"UPSERT\",
      \"ResourceRecordSet\": {
        \"Name\": \"${DOMAIN}\",
        \"Type\": \"A\",
        \"TTL\": 300,
        \"ResourceRecords\": [{\"Value\": \"${ELASTIC_IP}\"}]
      }
    }]
  }" >/dev/null

echo ""
echo "== Done =="
echo "Instance: $INSTANCE_ID"
echo "Elastic IP: $ELASTIC_IP"
echo "DNS: https://${DOMAIN} (allow a few minutes to propagate; Caddy needs this resolving before it can get a cert)"
echo ""
echo "Next: copy the app onto the instance, write its .env, and bring the stack up (see deploy/redeploy.sh)."
