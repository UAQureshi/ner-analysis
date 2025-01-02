import spacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
import networkx as nx
from spacy import displacy
from IPython.display import HTML
import json

def load_models():
    """Load required NLP models"""
    nlp = spacy.load("en_core_web_sm")
    ner_transformer = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
    return nlp, ner_transformer

def visualize_entities(doc, output_path=None):
    """Generate visualization for spaCy entities"""
    html = displacy.render(doc, style="ent")
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(create_html_wrapper(html, "spaCy NER Results"))
    return html

def create_transformer_visualization(text, transformer_results, output_path=None):
    """Create HTML visualization for transformer NER results"""
    # Create spans for visualization
    spans = []
    used_positions = set()
    
    # Sort results by confidence score
    sorted_results = sorted(transformer_results, key=lambda x: x['score'], reverse=True)
    
    for result in sorted_results:
        start = text.find(result['word'])
        if start == -1:  # Skip if word not found
            continue
            
        end = start + len(result['word'])
        
        # Check for overlap with existing spans
        overlap = False
        for pos in range(start, end):
            if pos in used_positions:
                overlap = True
                break
        
        if not overlap:
            spans.append({
                'start': start,
                'end': end,
                'label': result['entity'],
                'confidence': f"{result['score']:.2%}"
            })
            # Mark positions as used
            used_positions.update(range(start, end))
    
    # Generate HTML
    html = create_entity_html(text, spans)
    
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(create_html_wrapper(html, "Transformer NER Results"))
    
    return html

def create_entity_html(text, spans):
    """Create HTML markup for entity visualization"""
    html_parts = []
    last_end = 0
    
    # Sort spans by start position
    spans = sorted(spans, key=lambda x: x['start'])
    
    for span in spans:
        # Add text before entity
        html_parts.append(text[last_end:span['start']])
        
        # Add entity with highlighting
        entity_text = text[span['start']:span['end']]
        html_parts.append(
            f'<mark class="entity" style="background: {get_entity_color(span["label"])}; '
            f'padding: 0.2em 0.3em; margin: 0 0.2em; line-height: 1; border-radius: 0.35em;">'
            f'{entity_text}'
            f'<span style="font-size: 0.8em; font-weight: bold; line-height: 1; '
            f'border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">'
            f'{span["label"]} ({span["confidence"]})</span></mark>'
        )
        
        last_end = span['end']
    
    # Add remaining text
    html_parts.append(text[last_end:])
    
    return ''.join(html_parts)

def get_entity_color(entity_type):
    """Get color for entity type"""
    colors = {
        'PER': '#fca503',   # Orange for Person
        'ORG': '#7aecec',   # Light blue for Organization
        'LOC': '#ff9561',   # Salmon for Location
        'MISC': '#bfeeb7',  # Light green for Miscellaneous
        'DATE': '#9cc9cc',  # Blue-grey for Date
        'TIME': '#9cc9cc',  # Blue-grey for Time
        'MONEY': '#e4e7d2', # Light grey for Money
        'PERCENT': '#e4e7d2', # Light grey for Percent
    }
    return colors.get(entity_type, '#ddd')

def create_html_wrapper(content, title):
    """Wrap content in HTML with styling"""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                line-height: 1.5;
                padding: 2rem;
                max-width: 1200px;
                margin: 0 auto;
            }}
            .entity {{
                display: inline-block;
                border-radius: 0.35em;
                margin: 0.2em;
            }}
            .comparison {{
                display: flex;
                flex-direction: column;
                gap: 2rem;
            }}
            h2 {{
                color: #2a2a2a;
                border-bottom: 2px solid #eee;
                padding-bottom: 0.5rem;
            }}
        </style>
    </head>
    <body>
        <h2>{title}</h2>
        <div class="content">
            {content}
        </div>
    </body>
    </html>
    """

def plot_confidence_scores(df_transformer, output_path=None):
    """Plot confidence scores for transformer predictions"""
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_transformer, x='word', y='score')
    plt.xticks(rotation=45)
    plt.title('Entity Detection Confidence Scores')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.show() 