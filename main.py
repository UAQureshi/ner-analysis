import os
from pathlib import Path
import logging
from src.utils import load_models
from src.ner_processor import NERProcessor
from src.knowledge_graph import KnowledgeGraphBuilder
from src.document_chunker import EntityAwareChunker
from src.utils import visualize_entities, plot_confidence_scores
from src.utils import create_transformer_visualization, create_html_wrapper
import pandas as pd

def setup_logging():
    """Setup logging configuration"""
    results_dir = Path("results")
    log_dir = results_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'ner_analysis.log'),
            logging.StreamHandler()  # This will print to console
        ]
    )

def setup_directories():
    """Create necessary directories for results"""
    results_dir = Path("results")
    subdirs = ["graphs", "html", "plots", "analysis", "logs"]
    
    for subdir in subdirs:
        (results_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    return results_dir

def read_input_text(file_path: str = "input.txt") -> str:
    """Read input text from file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            logging.info(f"Successfully read input text from {file_path}")
            return text
    except FileNotFoundError:
        error_msg = f"Input file '{file_path}' not found. Please create it with your text content."
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)

def save_analysis_results(comparison_df, results_dir: Path):
    """Save analysis results to file and log them"""
    analysis_path = results_dir / "analysis/comparison_results.txt"
    
    # Calculate statistics
    total_entities = len(comparison_df)
    matching_entities = comparison_df.notna().all(axis=1).sum()
    spacy_only = comparison_df['spaCy'].notna().sum() - matching_entities
    transformer_only = comparison_df['Transformer'].notna().sum() - matching_entities
    
    # Log statistics
    logging.info("\n=== NER Model Comparison Analysis ===")
    logging.info(f"Total Unique Entities: {total_entities}")
    logging.info(f"Entities Found by Both Models: {matching_entities}")
    logging.info(f"Entities Found Only by spaCy: {spacy_only}")
    logging.info(f"Entities Found Only by Transformer: {transformer_only}")
    
    # Save to file
    with open(analysis_path, 'w', encoding='utf-8') as f:
        f.write("=== NER Model Comparison Analysis ===\n\n")
        f.write(f"Overall Statistics:\n")
        f.write(f"Total Unique Entities: {total_entities}\n")
        f.write(f"Entities Found by Both Models: {matching_entities}\n")
        f.write(f"Entities Found Only by spaCy: {spacy_only}\n")
        f.write(f"Entities Found Only by Transformer: {transformer_only}\n\n")
        
        f.write("Detailed Comparison:\n")
        f.write(comparison_df.to_string())
        
        # Analyze disagreements
        disagreements = comparison_df[comparison_df['spaCy'] != comparison_df['Transformer']]
        if not disagreements.empty:
            f.write("\n\nEntity Classification Disagreements:\n")
            f.write(disagreements.to_string())
            logging.info(f"\nFound {len(disagreements)} entity classification disagreements")

def main():
    # Setup logging and directories
    setup_logging()
    results_dir = setup_directories()
    logging.info("Starting NER analysis")
    
    # Read input text
    text = read_input_text()
    
    # Load models
    logging.info("Loading NLP models")
    nlp, ner_transformer = load_models()
    
    # Initialize processors
    ner_processor = NERProcessor(nlp, ner_transformer)
    graph_builder = KnowledgeGraphBuilder()
    chunker = EntityAwareChunker(nlp)
    
    # Process with spaCy
    logging.info("Processing text with spaCy")
    spacy_df, doc = ner_processor.process_with_spacy(text)
    logging.info(f"SpaCy found {len(spacy_df)} entities")
    
    # Process with transformer
    logging.info("Processing text with transformer")
    transformer_df = ner_processor.process_with_transformer(text)
    logging.info(f"Transformer found {len(transformer_df)} entities")
    
    # Compare and save results
    logging.info("Comparing results from both models")
    comparison_df = ner_processor.compare_results(text)
    save_analysis_results(comparison_df, results_dir)
    
    # Create visualizations
    logging.info("Creating visualizations")
    visualize_entities(doc, results_dir / "html/spacy_ner_results.html")
    create_transformer_visualization(
        text, 
        transformer_df.to_dict('records'),
        results_dir / "html/transformer_ner_results.html"
    )
    
    # Create combined visualization
    logging.info("Creating combined visualization")
    combined_html = f"""
    <div class="comparison">
        <div class="spacy-results">
            <h2>spaCy NER Results</h2>
            {visualize_entities(doc)}
        </div>
        <div class="transformer-results">
            <h2>Transformer NER Results</h2>
            {create_transformer_visualization(text, transformer_df.to_dict('records'))}
        </div>
    </div>
    """
    
    with open(results_dir / "html/combined_ner_results.html", "w", encoding="utf-8") as f:
        f.write(create_html_wrapper(combined_html, "NER Comparison"))
    
    # Plot confidence scores
    logging.info("Plotting confidence scores")
    plot_confidence_scores(
        transformer_df, 
        results_dir / "plots/confidence_scores.png"
    )
    
    # Create knowledge graphs
    logging.info("Creating knowledge graphs")
    G_contextual = graph_builder.create_contextual_graph(doc, min_weight=1)
    graph_builder.visualize_graph(
        G_contextual,
        results_dir / "graphs/contextual_knowledge_graph.png",
        "Contextual Knowledge Graph"
    )
    
    G_semantic = graph_builder.create_semantic_graph(doc)
    graph_builder.visualize_graph(
        G_semantic,
        results_dir / "graphs/semantic_knowledge_graph.png",
        "Semantic Knowledge Graph"
    )
    
    # Save chunking results
    logging.info("Processing document chunks")
    chunks = chunker.chunk_document(text)
    with open(results_dir / "analysis/document_chunks.txt", "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"\nChunk {i+1}:\n")
            f.write(chunk + "\n")
            entities = chunker.get_chunk_entities(chunk)
            f.write(f"Entities: {entities}\n")
    
    logging.info("NER analysis completed successfully")

if __name__ == "__main__":
    main() 