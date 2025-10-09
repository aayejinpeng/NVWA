#!/bin/bash
set -e

# -----------------------
# 配置
# -----------------------
USER="orangepi"
HOST="10.156.120.55"
PORT=10022
SSH_KEY="$HOME/.ssh/id_rsa"
REMOTE_BASE="/home/orangepi/nvwa"  # 远程基目录
LOG_FILE="./.log/benchmark_$(date +%Y%m%d_%H%M%S).log"

mkdir -p .log

# -----------------------
# 参数解析
# -----------------------
if [ "$1" == "-a" ]; then
    FOLDERS=(*/)
else
    if [ -z "$1" ]; then
        echo "Usage: $0 [-a] [folder_name]"
        exit 1
    fi
    FOLDERS=("$1")
fi

# -----------------------
# SSH可用性检查
# -----------------------
echo ">>> Checking SSH connectivity to $HOST..."
ssh -i "$SSH_KEY" -p "$PORT" -q -o BatchMode=yes -o ConnectTimeout=5 "$USER@$HOST" exit
if [ $? -ne 0 ]; then
    echo "ERROR: Cannot SSH to $HOST. Check network and credentials."
    exit 1
fi
echo "SSH connection successful."

# -----------------------
# 循环处理每个文件夹
# -----------------------
for LOCAL_DIR in "${FOLDERS[@]}"; do
    # 去掉末尾斜杠
    LOCAL_DIR="${LOCAL_DIR%/}"
    REMOTE_DIR="$REMOTE_BASE/$LOCAL_DIR"

    # source ./.venv/bin/activate

    echo ">>> Syncing $LOCAL_DIR to $HOST:$REMOTE_DIR ..."
    rsync -avz -e "ssh -i $SSH_KEY -p $PORT" \
        --exclude='.build/' \
        --exclude='__pycache__/' \
        "$LOCAL_DIR/" "$USER@$HOST:$REMOTE_DIR/"

    echo ">>> Executing bench.py in $REMOTE_DIR ..."
    ssh -i "$SSH_KEY" -p "$PORT" "$USER@$HOST" 2>&1 <<EOF | tee -a "$LOG_FILE"
cd $REMOTE_DIR || exit 1
source ../.venv/bin/activate || exit 1
python3 bench.py || exit 1
EOF

    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        echo "ERROR: bench.py in $LOCAL_DIR failed. Check log: $LOG_FILE"
        exit 1
    fi
done

echo ">>> All benchmarks completed. Log saved to $LOG_FILE"
