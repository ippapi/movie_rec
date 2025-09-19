#!/bin/bash

# ====== Config ======
STORAGE_ACCOUNTS=("ippapi" "ippapiindo" "ippapieast")
CONTAINER_NAME="training-data"
LOCAL_DIR="/home/chauhb/Projects/movie_rec/storage/training_data"
TMP_DIR="/tmp/blob_dowloads"
FILES=("train.parquet" "dev.parquet" "test.parquet")
CONN_VARS=("AZURE_CONNECTION_STRING_1" "AZURE_CONNECTION_STRING_2" "AZURE_CONNECTION_STRING_3")
# ====================

mkdir -p $TMP_DIR

# ======== Upload từng file và đo thời gian ========
for i in "${!STORAGE_ACCOUNTS[@]}"; do
    ACCOUNT=${STORAGE_ACCOUNTS[$i]}
    VAR_NAME=${CONN_VARS[$i]}
    AZURE_CONN=${!VAR_NAME}
    echo "===== Using storage account: $ACCOUNT ====="

    # Tạo container nếu chưa có
    echo "Checking/creating container..."
    az storage container create \
        --connection-string "$AZURE_CONN" \
        --name $CONTAINER_NAME \
        --account-name $ACCOUNT \
        --public-access off \

    # ---- Upload từng file và đo latency ----
    for FILE in "${FILES[@]}"; do
        echo "Uploading $FILE..."
        START=$(date +%s%3N)
        az storage blob upload \
            --connection-string "$AZURE_CONN" \
            --account-name $ACCOUNT \
            --container-name $CONTAINER_NAME \
            --name "$FILE" \
            --file "$LOCAL_DIR/$FILE" \
            --overwrite true
        END=$(date +%s%3N)
        LATENCY=$((END-START))
        echo "Upload $FILE done in $LATENCY ms"

        # ---- Download blob về tmp và đo latency ----
        TMP_FILE="$TMP_DIR/$FILE"
        echo "Downloading $FILE..."
        START=$(date +%s%3N)
        az storage blob download \
            --connection-string "$AZURE_CONN" \
            --account-name $ACCOUNT \
            --container-name $CONTAINER_NAME \
            --name "$FILE" \
            --file "$TMP_FILE" \
            --overwrite true
        END=$(date +%s%3N)
        LATENCY=$((END-START))
        echo "Download $FILE done in $LATENCY ms"

        # ---- Xóa file tạm ----
        rm -f "$TMP_FILE"
        echo "$FILE removed from tmp local"
    done

    echo "===== Done with $ACCOUNT ====="
done