#!/bin/bash
# Quick start scripts for PPO training

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}PPO Training Scripts for SUMO${NC}"
echo -e "${BLUE}=====================================${NC}\n"

# Function to print menu
print_menu() {
    echo -e "${YELLOW}Choose an option:${NC}"
    echo "1. Train PPO (Quick Test - 10 iterations)"
    echo "2. Train PPO (Medium - 100 iterations)"
    echo "3. Train PPO (Full - 500 iterations)"
    echo "4. Evaluate Trained Model"
    echo "5. Train with Custom Parameters"
    echo "6. View Results"
    echo "7. Exit"
    echo ""
}

# Function to train quick
train_quick() {
    echo -e "${GREEN}Starting quick training (10 iterations)...${NC}\n"
    python scripts/train_ppo.py \
        --network grid4x4 \
        --iterations 10 \
        --workers 1 \
        --checkpoint-interval 5 \
        --seed 42
}

# Function to train medium
train_medium() {
    echo -e "${GREEN}Starting medium training (100 iterations)...${NC}\n"
    python scripts/train_ppo.py \
        --network grid4x4 \
        --iterations 100 \
        --workers 2 \
        --checkpoint-interval 10 \
        --seed 42
}

# Function to train full
train_full() {
    echo -e "${GREEN}Starting full training (500 iterations)...${NC}\n"
    python scripts/train_ppo.py \
        --network zurich \
        --iterations 500 \
        --workers 4 \
        --checkpoint-interval 20 \
        --seed 42
}

# Function for custom parameters
train_custom() {
    echo -e "${YELLOW}Enter custom parameters:${NC}"
    read -p "Network name (grid4x4/4x4loop/zurich/PhuQuoc) [grid4x4]: " network
    network=${network:-grid4x4}
    
    read -p "Number of iterations [100]: " iterations
    iterations=${iterations:-100}
    
    read -p "Number of workers [2]: " workers
    workers=${workers:-2}
    
    read -p "Use GPU? (y/n) [n]: " gpu
    gpu_flag=""
    if [ "$gpu" == "y" ]; then
        gpu_flag="--gpu"
    fi
    
    echo -e "${GREEN}Starting training with custom parameters...${NC}\n"
    python scripts/train_ppo.py \
        --network $network \
        --iterations $iterations \
        --workers $workers \
        --checkpoint-interval 10 \
        $gpu_flag
}

# Function to evaluate
evaluate() {
    echo -e "${YELLOW}Enter checkpoint path:${NC}"
    read -p "Checkpoint path: " checkpoint
    
    if [ -z "$checkpoint" ]; then
        echo -e "${YELLOW}Using latest checkpoint from results...${NC}"
        checkpoint=$(ls -td results/ppo_*/ | head -n1)
        checkpoint="${checkpoint}checkpoint_000010"
    fi
    
    if [ ! -d "$checkpoint" ]; then
        echo -e "${RED}Checkpoint not found: $checkpoint${NC}"
        return
    fi
    
    read -p "Number of episodes [5]: " episodes
    episodes=${episodes:-5}
    
    read -p "Use GUI? (y/n) [n]: " gui
    gui_flag=""
    if [ "$gui" == "y" ]; then
        gui_flag="--gui"
    fi
    
    echo -e "${GREEN}Starting evaluation...${NC}\n"
    python scripts/eval_ppo.py \
        --checkpoint "$checkpoint" \
        --episodes $episodes \
        $gui_flag
}

# Function to view results
view_results() {
    echo -e "${GREEN}Recent training results:${NC}\n"
    if [ -d "results" ]; then
        ls -ldt results/ppo_*/ | head -10 | awk '{print $9, "(" $6, $7, $8 ")"}'
        echo ""
    else
        echo "No results found."
    fi
}

# Main loop
while true; do
    print_menu
    read -p "Enter your choice [1-7]: " choice
    
    case $choice in
        1) train_quick ;;
        2) train_medium ;;
        3) train_full ;;
        4) evaluate ;;
        5) train_custom ;;
        6) view_results ;;
        7) 
            echo -e "${GREEN}Goodbye!${NC}"
            exit 0
            ;;
        *)
            echo -e "${YELLOW}Invalid option. Please try again.${NC}\n"
            ;;
    esac
    
    echo ""
    read -p "Press Enter to continue..."
    clear
done
