#!/bin/bash

MODE=""
READ_SIZE="32768"
BATCH_COUNT=16
ITERATIONS=10000

usage() {
    echo "Usage: $0 -m <mode> [-s read_size] [-i <iterations>]"
    echo ""
    echo "  -m    Mode: performance or accuracy (required)"
    echo "  -s    Read size: Number of bytes read per iteration (default: 32768)"
    echo "  -i    Iterations (default: 10000)"
    echo "  -h    Show this help message"
    exit 1
}

while getopts ":m:s:i:h" opt; do
    case ${opt} in
        m)
            MODE=$OPTARG
            ;;
        s)
            READ_SIZE=$OPTARG
            ;;
        i)
            ITERATIONS=$OPTARG
            ;;
        h)
            usage
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            usage
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            usage
            ;;
    esac
done

if [[ "$MODE" != "performance" && "$MODE" != "accuracy" ]]; then
    echo "Error: mode must be 'performance' or 'accuracy'" >&2
    usage
fi

echo "--------------------------------------"
echo "MODE:        $MODE"
echo "READ_SIZE:   $READ_SIZE"
echo "ITERATIONS:  $ITERATIONS"
echo "--------------------------------------"

# Loop over FFT sizes (from 16 to 1024)
for I in $(seq 4 10); do
    FFT_SIZE=$((2 ** I))

    if [[ "$MODE" == "performance" ]]; then
        BATCH_COUNT=$(((READ_SIZE / 2) / FFT_SIZE))
    fi
    echo ""
    echo "FFT_SIZE:    $FFT_SIZE"
    echo "BATCH_COUNT: $BATCH_COUNT"
    
    ./test.py "$MODE" -s "$FFT_SIZE" -b "$BATCH_COUNT" -i "$ITERATIONS"
done
