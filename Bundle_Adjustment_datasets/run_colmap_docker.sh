#!/bin/bash

# ==============================================================================
# COLMAP BENCHMARK SCRIPT: Refactored for Robust Path Handling (Absolute Paths)
# ==============================================================================
# TARGET: Process 1DSfM datasets (Montreal, Alamo, Roman Forum, Piccadilly)
# STRATEGY: Use Exhaustive Matcher for small datasets, Vocab Tree for large.
# FIXES: Corrected path handling, updated FAISS vocabulary filename, fixed
#        ImageReader.single_camera error.
# ==============================================================================

# --- 1. CONFIGURATION ---
# ------------------------

# IMPORTANT: REPLACE THIS with the verified absolute path to your '1dsfm' root folder.
# Example: /home/yourusername/GTSAM-Project/Bundle_Adjustment_datasets/1dsfm
COLMAP_DATA_ROOT="/home/anatharv1/GTSAM-Project/Bundle_Adjustment_datasets/1dsfm" 

# FIX: Use the exact, verified filename for the FAISS-compatible vocabulary tree.
VOCAB_TREE_HOST_PATH="${COLMAP_DATA_ROOT}/vocab_tree_faiss_flickr100K_words256K.bin"

# The specific datasets to process.
TARGET_DATASETS=("Alamo" "Roman_Forum" "Piccadilly" "Trafalgar")

# Docker Image (We assume the latest is used, which supports FAISS)
DOCKER_IMAGE="colmap/colmap"
DOCKER_VOCAB_TREE_PATH="/docker/vocab_tree.bin" # Consistent path inside the container

# --- 2. UTILITY FUNCTIONS ---
# ----------------------------

# Function to resolve the absolute path for robustness
function resolve_path() {
    local path="$1"
    if [[ "$path" == ~* ]]; then
        echo "${path/\~/$HOME}"
    else
        echo "$path"
    fi
}

# Function to display error message and exit
function die() {
    echo -e "\n❌ ERROR: $1" >&2
    exit 1
}

# --- 3. PRE-FLIGHT CHECKS ---
# ----------------------------

# Set the absolute root path using the utility function
COLMAP_DATA_ROOT=$(resolve_path "$COLMAP_DATA_ROOT")

# Validate the main root directory
if [ ! -d "$COLMAP_DATA_ROOT" ]; then
    die "COLMAP Data Root directory not found: ${COLMAP_DATA_ROOT}"
fi

# Validate the vocabulary tree (Crucial for fast matching)
if [ ! -f "$VOCAB_TREE_HOST_PATH" ]; then
    echo "⚠️ WARNING: FAISS Vocabulary tree not found at ${VOCAB_TREE_HOST_PATH}"
    echo "  Vocab Tree matching will be DISABLED. Large datasets will use the slow Exhaustive Matcher."
    VOCAB_TREE_HOST_PATH=""
fi

echo "========================================================================"
echo "STARTING FAST RECONSTRUCTION PIPELINE"
echo "Target Datasets: ${TARGET_DATASETS[*]}"
echo "Data Root: ${COLMAP_DATA_ROOT}"
echo "========================================================================"

# --- 4. CORE PROCESSING LOOP ---
# -------------------------------

for dataset_name in "${TARGET_DATASETS[@]}"; do
    
    # Define HOST paths explicitly using the root
    HOST_DATASET_DIR="${COLMAP_DATA_ROOT}/${dataset_name}"
    HOST_IMAGE_DIR="${COLMAP_DATA_ROOT}/images.${dataset_name}/${dataset_name}/images"

    echo ""
    echo "----------------------------------------------------------------"
    echo "Processing: $dataset_name"

    # --- Validation and Setup ---
    if [ ! -d "$HOST_DATASET_DIR" ]; then
        echo "[SKIP] Data directory not found: $HOST_DATASET_DIR"
        continue
    fi
    if [ ! -d "$HOST_IMAGE_DIR" ]; then
        echo "[SKIP] Image source directory not found: $HOST_IMAGE_DIR"
        continue
    fi

    # Count images and determine strategy
    num_images=$(ls -1 "$HOST_IMAGE_DIR" | wc -l)
    echo "Image Count: $num_images"

    # Define paths inside the Docker container
    DOCKER_WORKSPACE="/workspace"
    DOCKER_IMAGE_PATH="/images"
    DOCKER_DATABASE_PATH="${DOCKER_WORKSPACE}/database.db"

    # Strategy Selection (500 images is the standard cutoff)
    if [ "$num_images" -lt 500 ]; then
        echo "Strategy: SMALL (<500) -> Using Exhaustive Matcher"
        MATCHER_CMD="colmap exhaustive_matcher --database_path ${DOCKER_DATABASE_PATH} --SiftMatching.use_gpu 1"
    elif [ -f "$VOCAB_TREE_HOST_PATH" ]; then
        echo "Strategy: LARGE (>=500) -> Using Vocab Tree Matcher (FAISS)"
        MATCHER_CMD="colmap vocab_tree_matcher --database_path ${DOCKER_DATABASE_PATH} --VocabTreeMatching.vocab_tree_path ${DOCKER_VOCAB_TREE_PATH} --SiftMatching.use_gpu 1 --SiftMatching.num_threads 8"
    else
        echo "Strategy: LARGE (>=500) -> FALLBACK to Exhaustive Matcher (Vocab tree missing)"
        MATCHER_CMD="colmap exhaustive_matcher --database_path ${DOCKER_DATABASE_PATH} --SiftMatching.use_gpu 1"
    fi

# --- RUN DOCKER EXECUTION ---
    echo "Starting Docker container..."
    
    docker run \
        --rm \
        --gpus all \
        -v "$HOST_DATASET_DIR:${DOCKER_WORKSPACE}" \
        -v "$HOST_IMAGE_DIR:${DOCKER_IMAGE_PATH}" \
        $( [ -n "$VOCAB_TREE_HOST_PATH" ] && echo "-v $VOCAB_TREE_HOST_PATH:$DOCKER_VOCAB_TREE_PATH" ) \
        -w ${DOCKER_WORKSPACE} \
        --entrypoint /bin/bash \
        "$DOCKER_IMAGE" \
        -c "
            # Set the execution steps inside the container
            
            # Remove old database for a clean start
            rm -f ${DOCKER_DATABASE_PATH}

            echo '>>> [1/4] Feature Extraction (Fast Mode)...' && \
            colmap feature_extractor \
                --database_path ${DOCKER_DATABASE_PATH} \
                --image_path ${DOCKER_IMAGE_PATH} \
                --ImageReader.single_camera 0 \
                --SiftExtraction.max_image_size 1200 \
                --SiftExtraction.max_num_features 4096 \
                --SiftExtraction.use_gpu 1 \
                || exit 1 ; \
            \
            echo '>>> [2/4] Matching Features...' && \
            ${MATCHER_CMD} \
                || exit 1 ; \
            \
            echo '>>> [3/4] Sparse Reconstruction (Fast BA)...' && \
            mkdir -p sparse && \
            colmap mapper \
                --database_path ${DOCKER_DATABASE_PATH} \
                --image_path ${DOCKER_IMAGE_PATH} \
                --output_path sparse \
                --Mapper.ba_global_function_tolerance 0.000001 \
                --Mapper.ba_global_max_num_iterations 20 \
                || exit 1 ; \
            \
            echo '>>> [4/4] Converting to Bundler Format...' && \
            colmap model_converter \
                --input_path sparse/0 \
                --output_path colmap_bundle.out \
                --output_type Bundler \
                || exit 1
        " || die "Docker container execution failed for ${dataset_name}"

    # Final Check on the HOST machine
    if [ -f "${HOST_DATASET_DIR}/colmap_bundle.out" ]; then
        echo "[SUCCESS] Generated: ${HOST_DATASET_DIR}/colmap_bundle.out"
    else
        echo "[FAILURE] Output file not found for $dataset_name. Check Docker logs above."
    fi

done

echo "========================================================================"
echo "Batch Processing Complete."
echo "========================================================================"