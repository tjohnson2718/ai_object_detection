#!/bin/bash

# test_detection_server.sh - Script to test FastAPI object detection server endpoints

# Configuration
SERVER_URL="http://localhost:8000"
TEST_IMAGE="./data/test_image_street.jpg"
OUTPUT_DIR="./test_results"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Function to test an endpoint and log results
test_endpoint() {
    local endpoint=$1
    local method=$2
    local description=$3
    local data=$4
    local output_file="$OUTPUT_DIR/$(echo $endpoint | tr '/' '_')_$(date +%s).json"
    
    echo -e "${YELLOW}Testing: ${NC}$description"
    echo -e "${YELLOW}Endpoint: ${NC}$method $SERVER_URL$endpoint"
    
    if [ "$method" == "GET" ]; then
        response=$(curl -s -w "\n%{http_code}" "$SERVER_URL$endpoint")
    else
        response=$(curl -s -w "\n%{http_code}" -X "$method" \
            "$SERVER_URL$endpoint" \
            -H "Content-Type: application/json" \
            -d "$data")
    fi
    
    # Extract status code (last line) and response body
    status_code=$(echo "$response" | tail -n1)
    response_body=$(echo "$response" | sed '$d')
    
    # Save response to file
    echo "$response_body" > "$output_file"
    
    # Check status code
    if [[ $status_code -ge 200 && $status_code -lt 300 ]]; then
        echo -e "${GREEN}Success! Status: $status_code${NC}"
        echo "Response saved to: $output_file"
    else
        echo -e "${RED}Failed! Status: $status_code${NC}"
        echo "Response: $response_body"
    fi
    echo "------------------------------------"
}

# Function to encode an image to base64
encode_image() {
    local image_path=$1
    if [ -f "$image_path" ]; then
        base64_image=$(base64 -w 0 "$image_path")
        echo "$base64_image"
    else
        echo "Error: Image file not found: $image_path"
        exit 1
    fi
}

# Main testing function
run_tests() {
    echo -e "${GREEN}Starting API endpoint tests...${NC}"
    echo "Server URL: $SERVER_URL"
    echo "Test image: $TEST_IMAGE"
    echo "Output directory: $OUTPUT_DIR"
    echo "======================================"
    
    # Test 1: Health check
    test_endpoint "/health" "GET" "Health Check" ""
    
    # Test 2: Get available classes
    test_endpoint "/classes" "GET" "Get Available Classes" ""
    
    # Encode the test image
    if [ -f "$TEST_IMAGE" ]; then
        base64_image=$(encode_image "$TEST_IMAGE")
        timestamp=$(date +%s)
        
        # Test 3: Detect all objects
        json_data="{\"request\": {\"image_data\": \"$base64_image\", \"timestamp\": $timestamp}}"
        test_endpoint "/detect" "POST" "Detect All Objects" "$json_data"
        
        # Test 4: Detect specific objects (person, car)
        test_endpoint "/detect?custom_classes=person&custom_classes=car" "POST" \
            "Detect Specific Objects (person, car)" "$json_data"
        
        # Test 5: Detect with non-existent class
        test_endpoint "/detect?custom_classes=nonexistent" "POST" \
            "Detect Non-existent Class" "$json_data"
        
    else
        echo -e "${RED}Test image not found: $TEST_IMAGE${NC}"
        echo "Skipping image detection tests."
    fi

    echo -e "${GREEN}All tests completed!${NC}"
    echo "Results saved in: $OUTPUT_DIR"
}

# Run tests
run_tests