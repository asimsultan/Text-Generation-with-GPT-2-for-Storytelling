
# Text Generation with GPT-2 for Storytelling

Welcome to the Text Generation with GPT-2 for Storytelling project! This project focuses on generating short stories using the GPT-2 model.

## Introduction

Text generation involves creating text based on the context provided. In this project, we leverage the power of GPT-2 to generate short stories using a dataset of story prompts.

## Dataset

For this project, we will use a custom dataset of story prompts. You can create your own dataset and place it in the `data/story_data.csv` file.

## Project Overview

### Prerequisites

- Python 3.6 or higher
- PyTorch
- Hugging Face Transformers
- Datasets

### Installation

To set up the project, follow these steps:

```bash
# Clone this repository and navigate to the project directory:
git clone https://github.com/your-username/gpt2_storytelling.git
cd gpt2_storytelling

# Install the required packages:
pip install -r requirements.txt

# Ensure your data includes story prompts. Place these files in the data/ directory.
# The data should be in a CSV file with one column: text.

# To fine-tune the GPT-2 model for story generation, run the following command:
python scripts/train.py --data_path data/story_data.csv

# To evaluate the performance of the fine-tuned model, run:
python scripts/evaluate.py --model_path models/ --data_path data/story_data.csv
