#!/bin/bash

# ====== Config ======
STORAGE_ACCOUNTS=("ippapi")
CONTAINER_NAME="prepared-data"
LOCAL_DIR="/home/chauhb/Projects/movie_rec/storage/prepared_data"
FILES=("movie_df.parquet" "users_movies.parquet")
CONN_VARS=("AZURE_CONNECTION_STRING_1")
# ====================

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
    done

    echo "===== Done with $ACCOUNT ====="
done