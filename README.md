# Named Entity Recognition (NER) Analysis Tool

A comprehensive tool for analyzing and comparing different Named Entity Recognition approaches, with a focus on applications in Retrieval Augmented Generation (RAG) systems.

## Overview

This tool provides a complete pipeline for:
- Named Entity Recognition using multiple approaches (spaCy and Transformers)
- Comparative analysis of different NER models
- Knowledge graph generation from extracted entities
- Entity-aware document chunking
- Detailed visualizations and analysis reports

## Features

- Multiple NER approaches:
  - spaCy (fast, rule-based + statistical)
  - Transformer-based (BERT model, higher accuracy)
- Knowledge Graph Generation:
  - Contextual relationships
  - Semantic relationships
- Visualizations:
  - Interactive HTML entity highlighting
  - Confidence score plots
  - Knowledge graphs
- Analysis Reports:
  - Model comparison
  - Entity disagreements
  - Chunking results

## Requirements

```bash
# Core Dependencies
transformers==4.35.2
datasets==2.14.5
torch==2.1.0
spacy==3.7.2
networkx==3.1
matplotlib==3.7.1
seaborn==0.12.2
pandas==2.0.3
ipython==8.12.0

# Install spaCy English model
python -m spacy download en_core_web_sm
```

## Project Structure

```
.
├── src/
│   ├── utils.py              # Utility functions
│   ├── ner_processor.py      # NER processing classes
│   ├── knowledge_graph.py    # Graph generation
│   └── document_chunker.py   # Document chunking
├── results/
│   ├── graphs/              # Knowledge graph visualizations
│   ├── html/                # NER visualization results
│   ├── plots/               # Confidence score plots
│   ├── analysis/            # Text analysis results
│   └── logs/                # Processing logs
├── input.txt                # Input text file
├── requirements.txt         # Project dependencies
└── main.py                 # Main execution script
```

## Usage

1. Prepare Input:
   ```bash
   # Create input.txt with your text content
   echo "Your text here..." > input.txt
   ```

2. Install Dependencies:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

3. Run Analysis:
   ```bash
   python main.py
   ```

## Output Files

### HTML Visualizations
- `results/html/spacy_ner_results.html`: spaCy entity highlighting
- `results/html/transformer_ner_results.html`: Transformer entity highlighting
- `results/html/combined_ner_results.html`: Side-by-side comparison

### Graphs
- `results/graphs/contextual_knowledge_graph.png`: Entity relationships based on co-occurrence
- `results/graphs/semantic_knowledge_graph.png`: Entity relationships based on semantic rules

### Analysis
- `results/analysis/comparison_results.txt`: Detailed model comparison
- `results/analysis/document_chunks.txt`: Entity-aware document chunks
- `results/plots/confidence_scores.png`: Entity detection confidence visualization

### Logs
- `results/logs/ner_analysis.log`: Processing logs and statistics

## Input Format

The tool accepts plain text input through `input.txt`. The text should be:
- UTF-8 encoded
- One or more paragraphs
- No special formatting required

Example:
```text
On January 20, 2023, Elon Musk, the CEO of Tesla Inc., announced...
```

## Results Interpretation

### NER Comparison
- Compares entities found by spaCy and Transformer models
- Shows agreements and disagreements in entity classification
- Provides confidence scores for transformer predictions

### Knowledge Graphs
1. Contextual Graph:
   - Shows entity relationships based on co-occurrence
   - Edge weights indicate frequency of co-occurrence
   - Node colors represent entity types

2. Semantic Graph:
   - Shows relationships based on predefined semantic rules
   - Helps understand entity interactions
   - Useful for knowledge base construction

### Document Chunking
- Demonstrates entity-aware text splitting
- Useful for RAG applications
- Preserves entity context

## Contributing

Feel free to contribute by:
1. Opening issues for bugs or feature requests
2. Submitting pull requests
3. Improving documentation

## License

MIT License - feel free to use and modify for your needs.

## Acknowledgments

- spaCy for their excellent NLP library
- Hugging Face for transformer models
- NetworkX for graph visualization 