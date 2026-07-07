#!/bin/bash

set -e

echo "=============================="
echo "Q-Stream Full Pipeline"
echo "Running 4.1 to 4.7"
echo "=============================="

cd /root/autodl-tmp/Q_Stream

echo ""
echo "Checking input video..."
if [ ! -f "sample_video.mp4" ]; then
    echo "Error: sample_video.mp4 not found."
    echo "Please put your video in Q_Stream and name it sample_video.mp4"
    exit 1
fi

file sample_video.mp4

echo ""
echo "Cleaning old outputs..."
rm -rf output_features
rm -rf output_events
rm -rf output_query_forecasts
rm -rf output_memory
rm -rf output_answers

echo ""
echo "4.2 + 4.3: Running Streaming Frame Sampling and Frame Encoding..."
export HF_ENDPOINT=https://hf-mirror.com
python pipeline_stage1.py

echo ""
echo "4.4: Running Event Construction..."
python event_construction.py

echo ""
echo "4.5: Running Query Forecasting..."
python query_forecasting.py

echo ""
echo "4.6: Running Query-Aware Memory Management..."
python memory_management.py

echo ""
echo "4.7: Running Evidence-Grounded Answering..."
echo "What happened in the video?" | python answering.py

echo ""
echo "=============================="
echo "Full pipeline completed."
echo "Generated outputs:"
echo "=============================="

echo ""
echo "output_features:"
ls output_features

echo ""
echo "output_events:"
ls output_events

echo ""
echo "output_query_forecasts:"
ls output_query_forecasts

echo ""
echo "output_memory:"
ls output_memory

echo ""
echo "output_answers:"
ls output_answers