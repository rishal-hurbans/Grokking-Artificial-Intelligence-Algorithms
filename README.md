# Grokking AI Algorithms

[Get Grokking Artificial Intelligence Algorithms at Manning Publications](https://www.manning.com/books/grokking-artificial-intelligence-algorithms?a_aid=gaia&a_bid=6a1b836a) 

Rather Learn by exploring the code notebook in your browser? Click here:

[![Open the interactive code notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rishal-hurbans/Grokking-Artificial-Intelligence-Algorithms-Notebook/blob/main/Grokking_Artificial_Intelligence_Algorithms_Notebook.ipynb)

## Requirements
* Python 3.9 or later (3.11 recommended)
* pip 23.1+ (comes with recent Python installers)
* Optional: PyTorch-compatible GPU for the heavy demos in Chapters 11–12

## Setup
1. **Install Python** – download the latest 3.x release from [python.org](https://www.python.org/downloads/) or use your platform package manager. Ensure `python` and `pip` point to the same interpreter (`python -m pip --version`).
2. **Create a virtual environment** (recommended so project dependencies stay isolated):
   * macOS / Linux:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   * Windows (PowerShell):
     ```powershell
     py -3 -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   *PyTorch wheels are large; on Apple Silicon use the `arm64` build from `pip` or follow the [official instructions](https://pytorch.org/get-started/locally/) if you need CUDA support.*
4. **Run an example** by moving into the chapter directory and executing the script:
   ```bash
   cd ch03-intelligent_search/informed_search
   python3 maze_astar.py
   ```

## Overview
This is the official supporting code for the book, Grokking AI Algorithms, published by Manning Publications, authored by Rishal Hurbans.

![History of AI](readme_assets/history_of_ai.png)

The example implementations provided will make more sense if you've read the book, however, it might be somewhat useful to you otherwise.

The purpose of this repository is to act as a practical reference for examples of how the algorithms mentioned in the book can be implemented.
This repository should not be consulted as the book is read page-by-page, but rather, when you're attempting to implement an algorithm or gain a more technical understanding from a programming perspective.

