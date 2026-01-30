#!/bin/bash
# ============================================
# UAV to Epona Dataset Conversion Script
# ============================================
# This script converts all UAV ROS bag datasets to Epona format
# with a single command.
#
# Usage:
#   ./run_convert.sh              # Use default config.yaml
#   ./run_convert.sh custom.yaml  # Use custom config file
#
# Prerequisites:
#   - ROS environment sourced (source /opt/ros/<distro>/setup.bash)
#   - Python packages: rosbag, cv_bridge, opencv-python, scipy, pyyaml, tqdm
# ============================================

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Default config file
CONFIG_FILE="${1:-${SCRIPT_DIR}/config.yaml}"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo "Usage: $0 [config.yaml]"
    exit 1
fi

# Check for ROS environment
if [ -z "$ROS_DISTRO" ]; then
    echo "Warning: ROS environment not detected."
    echo "Attempting to source ROS Noetic..."
    
    if [ -f "/opt/ros/noetic/setup.bash" ]; then
        source /opt/ros/noetic/setup.bash
        echo "Sourced ROS Noetic"
    elif [ -f "/opt/ros/melodic/setup.bash" ]; then
        source /opt/ros/melodic/setup.bash
        echo "Sourced ROS Melodic"
    else
        echo "Error: Could not find ROS installation."
        echo "Please source your ROS environment manually:"
        echo "  source /opt/ros/<distro>/setup.bash"
        exit 1
    fi
fi

echo "============================================"
echo "UAV to Epona Dataset Converter"
echo "============================================"
echo "ROS Distribution: $ROS_DISTRO"
echo "Config file: $CONFIG_FILE"
echo "Python: $(which python3)"
echo "============================================"

# Add ROS Python packages to PYTHONPATH
if [ -d "/opt/ros/${ROS_DISTRO}/lib/python3/dist-packages" ]; then
    export PYTHONPATH="/opt/ros/${ROS_DISTRO}/lib/python3/dist-packages:$PYTHONPATH"
    echo "Added ROS Python path to PYTHONPATH"
fi

# Check Python dependencies
echo "Checking Python dependencies..."
python3 -c "import rosbag" 2>/dev/null || {
    echo "Error: rosbag package not found."
    echo "Make sure ROS is properly installed and sourced."
    echo "Try: source /opt/ros/noetic/setup.bash"
    exit 1
}

python3 -c "import scipy" 2>/dev/null || {
    echo "Error: scipy package not found. Install with:"
    echo "  pip install scipy"
    exit 1
}

python3 -c "import cv2" 2>/dev/null || {
    echo "Error: opencv-python package not found. Install with:"
    echo "  pip install opencv-python"
    exit 1
}

python3 -c "import yaml" 2>/dev/null || {
    echo "Error: PyYAML package not found. Install with:"
    echo "  pip install pyyaml"
    exit 1
}

python3 -c "import tqdm" 2>/dev/null || {
    echo "Error: tqdm package not found. Install with:"
    echo "  pip install tqdm"
    exit 1
}

echo "All dependencies satisfied."
echo ""

# Run the conversion
echo "Starting batch conversion..."
echo "This will convert all UAV datasets to Epona format."
echo ""

PYTHONPATH="/opt/ros/${ROS_DISTRO}/lib/python3/dist-packages:$PYTHONPATH" \
    python3 "${SCRIPT_DIR}/batch_convert.py" "$@"

echo ""
echo "============================================"
echo "Conversion completed successfully!"
echo "============================================"
echo ""
echo "Next steps:"
echo "1. Check the output directory specified in config.yaml"
echo "2. Use the Epona config file: dit_config_dcae_uav.py"
echo "3. Run Epona inference:"
echo "   python3 scripts/test/test_nuplan.py \\"
echo "     --exp_name 'test-uav' \\"
echo "     --start_id 0 --end_id 1 \\"
echo "     --resume_path 'pretrained/epona_nuplan.pkl' \\"
echo "     --config configs/dit_config_dcae_uav.py"
echo "============================================"
