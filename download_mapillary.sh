#!/bin/bash
# Script to download Mapillary Vistas dataset
# 
# Usage:
#   1. Get your download URL from https://www.mapillary.com/dataset/vistas
#   2. Run: ./download_mapillary.sh "YOUR_DOWNLOAD_URL"

set -e

# Configuration
DOWNLOAD_DIR="/home/p24s09/image_angle_classifier/raw_data/Mapillary"
DOWNLOAD_URL="$1"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if URL is provided
if [ -z "$DOWNLOAD_URL" ]; then
    echo -e "${RED}Error: No download URL provided${NC}"
    echo ""
    echo "Steps to download Mapillary Vistas dataset:"
    echo "1. Go to: https://www.mapillary.com/dataset/vistas"
    echo "2. Log in with your Mapillary account (create one if needed)"
    echo "3. Accept the license terms"
    echo "4. Right-click on 'Download' button and copy the download link"
    echo "5. Run this script with the URL:"
    echo ""
    echo "   ./download_mapillary.sh \"YOUR_DOWNLOAD_URL\""
    echo ""
    exit 1
fi

# Create download directory
mkdir -p "$DOWNLOAD_DIR"
cd "$DOWNLOAD_DIR"

echo -e "${GREEN}Starting Mapillary Vistas download...${NC}"
echo "Download directory: $DOWNLOAD_DIR"
echo ""

# Download with progress bar
wget --continue \
     --progress=bar:force \
     --show-progress \
     --content-disposition \
     "$DOWNLOAD_URL"

# Check if download was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Download completed successfully!${NC}"
    echo ""
    echo "Files in $DOWNLOAD_DIR:"
    ls -lh "$DOWNLOAD_DIR"
    echo ""
    echo -e "${YELLOW}To extract the dataset, run:${NC}"
    echo "  cd $DOWNLOAD_DIR"
    echo "  unzip *.zip"
else
    echo -e "${RED}Download failed!${NC}"
    exit 1
fi
