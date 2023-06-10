#!/bin/bash

# Input directories
TRAINING_DATA_DIR="training_data"
TESTING_DATA_DIR="testing_data"

# Output directories
OUTPUT_DIR="output"

TRAINING_DATA_SEQ="${OUTPUT_DIR}/training_data_seq"
TESTING_DATA_SEQ="${OUTPUT_DIR}/testing_data_seq"
TRAINING_DATA_VECTORS="${OUTPUT_DIR}/training_data_vectors"
TESTING_DATA_VECTORS="${OUTPUT_DIR}/testing_data_vectors"

MODEL_DIR="${OUTPUT_DIR}/model"
LABELINDEX_DIR="${OUTPUT_DIR}/labelindex"
PREDICTIONS_DIR="${OUTPUT_DIR}/predictions"

# Convert training and testing dataset_sites to SequenceFile format
mahout seqdirectory \
   -i "${TRAINING_DATA_DIR}" \
   -o "${TRAINING_DATA_SEQ}" \
   -c UTF-8 -ow

mahout seqdirectory \
   -i "${TESTING_DATA_DIR}" \
   -o "${TESTING_DATA_SEQ}" \
   -c UTF-8 -ow

# Create term frequency-vectors for training and testing dataset_sites
mahout seq2sparse \
   -i "${TRAINING_DATA_SEQ}" \
   -o "${TRAINING_DATA_VECTORS}" \
   --maxDFPercent 85 --namedVector -ow

mahout seq2sparse \
   -i "${TESTING_DATA_SEQ}" \
   -o "${TESTING_DATA_VECTORS}" \
   --maxDFPercent 85 --namedVector -ow

# Train the Naive Bayes model
mahout trainnb \
   -i "${TRAINING_DATA_VECTORS}/tf-vectors" \
   -el \
   -o "${MODEL_DIR}" \
   -li "${LABELINDEX_DIR}" \
   -ow

# Test the model
mahout testnb \
   -i "${TESTING_DATA_VECTORS}/tf-vectors" \
   -m "${MODEL_DIR}" \
   -l "${LABELINDEX_DIR}" \
   -ow -o "${PREDICTIONS_DIR}"

# Evaluate the model (optional, you can use other evaluation tools)
echo "Check the content of ${PREDICTIONS_DIR} to evaluate the model's performance."
