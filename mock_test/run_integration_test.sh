#!/bin/bash

# Phase 3 Integration Test - One-Click Runner
# 一键运行完整的集成测试流程

set -e  # Exit on error

echo "============================================================"
echo "🚀 Phase 3 Integration Test - One-Click Runner"
echo "============================================================"
echo ""

# Configuration
MOCK_MODELS_DIR="./mock_models"
DEVICE="cpu"  # Change to "cuda" if GPU available

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Generate Mock Checkpoints
echo "============================================================"
echo "📦 Step 1: Generating Mock Checkpoints"
echo "============================================================"
echo ""

if [ -d "$MOCK_MODELS_DIR" ]; then
    echo -e "${YELLOW}⚠️  Mock models directory already exists: $MOCK_MODELS_DIR${NC}"
    read -p "Do you want to regenerate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$MOCK_MODELS_DIR"
        echo -e "${GREEN}✅ Removed existing mock models${NC}"
    else
        echo -e "${YELLOW}⏭️  Skipping checkpoint generation${NC}"
    fi
fi

if [ ! -d "$MOCK_MODELS_DIR" ]; then
    python mock_test/generate_mock_checkpoints.py --output-dir "$MOCK_MODELS_DIR"

    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✅ Step 1 Complete: Mock checkpoints generated${NC}"
    else
        echo ""
        echo -e "${RED}❌ Step 1 Failed: Could not generate mock checkpoints${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✅ Step 1 Complete: Using existing mock checkpoints${NC}"
fi

echo ""
echo "============================================================"
echo "🧪 Step 2: Running Integration Tests"
echo "============================================================"
echo ""

python mock_test/test_phase3_integration.py \
    --models-dir "$MOCK_MODELS_DIR" \
    --device "$DEVICE"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Step 2 Complete: All integration tests passed${NC}"
else
    echo ""
    echo -e "${RED}❌ Step 2 Failed: Integration tests failed${NC}"
    exit 1
fi

# Optional: Test real LIBERO integration
echo ""
echo "============================================================"
echo "🤖 Optional: Real LIBERO Integration Test"
echo "============================================================"
echo ""
echo "This test uses real LIBERO environment with Mock OpenVLA."
echo "Requires: LIBERO installed, MuJoCo, robosuite"
echo ""
read -p "Do you want to test real LIBERO integration? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "============================================================"
    echo "🤖 Step 3: Testing Real LIBERO Integration"
    echo "============================================================"
    echo ""

    # Check if LIBERO is installed
    if ! python -c "import libero" 2>/dev/null; then
        echo -e "${YELLOW}⚠️  LIBERO not found. Please install:${NC}"
        echo "   git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git"
        echo "   cd LIBERO && pip install -e ."
        echo ""
        echo -e "${YELLOW}⏭️  Skipping LIBERO test${NC}"
    else
        python mock_test/test_libero_with_mock_openvla.py \
            --models-dir "$MOCK_MODELS_DIR" \
            --device "$DEVICE" \
            --num-episodes 3 \
            --task-id 0

        if [ $? -eq 0 ]; then
            echo ""
            echo -e "${GREEN}✅ Step 3 Complete: Real LIBERO integration test passed${NC}"
        else
            echo ""
            echo -e "${RED}❌ Step 3 Failed: LIBERO integration test failed${NC}"
            echo "   This is a critical issue - fix before training on A100!"
            exit 1
        fi
    fi
fi

# Final Summary
echo ""
echo "============================================================"
echo "✅ All Tests Complete!"
echo "============================================================"
echo ""
echo "📊 Summary:"
echo "   • Mock checkpoints generated: $MOCK_MODELS_DIR"
echo "   • Integration tests: PASSED ✅"
echo "   • Device used: $DEVICE"
echo ""
echo "🎯 Next Steps:"
echo "   1. ✅ Local integration verified"
echo "   2. 📤 Deploy code to Modal"
echo "   3. 🔥 Train Phase 1 (Robust RFSQ)"
echo "   4. 🚀 Train Phase 2 (Draft + Main)"
echo "   5. 🤖 Run Phase 3 LIBERO evaluation"
echo ""
echo "📖 See AGENT_ACTION_PLAN.md for detailed instructions"
echo ""
