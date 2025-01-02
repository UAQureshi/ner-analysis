import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from collections import defaultdict
import spacy
from itertools import combinations

class KnowledgeGraphBuilder:
    def __init__(self):
        self.entity_colors = {
            'PERSON': '#fca503',    # Orange
            'ORG': '#7aecec',       # Light blue
            'GPE': '#ff9561',       # Salmon
            'LOC': '#ff9561',       # Salmon
            'DATE': '#9cc9cc',      # Blue-grey
            'MONEY': '#e4e7d2',     # Light grey
            'MISC': '#bfeeb7'       # Light green
        }

    def create_contextual_graph(self, doc, min_weight: int = 1) -> nx.Graph:
        """Create a knowledge graph based on sentence-level co-occurrence"""
        G = nx.Graph()
        edge_weights = defaultdict(int)
        
        # Process each sentence
        for sent in doc.sents:
            # Get entities in this sentence
            sent_entities = [(ent.text, ent.label_) for ent in sent.ents]
            
            # Add nodes with their types
            for entity, entity_type in sent_entities:
                if not G.has_node(entity):
                    G.add_node(entity, type=entity_type)
            
            # Add edges for co-occurring entities
            for (entity1, _), (entity2, _) in combinations(sent_entities, 2):
                if entity1 != entity2:
                    edge_weights[(entity1, entity2)] += 1
        
        # Add edges with weights above threshold
        for (entity1, entity2), weight in edge_weights.items():
            if weight >= min_weight:
                G.add_edge(entity1, entity2, weight=weight)
        
        return G

    def create_semantic_graph(self, doc) -> nx.Graph:
        """Create a knowledge graph based on semantic relationships"""
        G = nx.Graph()
        
        # Define semantic rules
        semantic_rules = {
            ('PERSON', 'ORG'): 'works_for',
            ('ORG', 'GPE'): 'located_in',
            ('PERSON', 'GPE'): 'located_in',
            ('ORG', 'ORG'): 'collaborates_with',
            ('GPE', 'GPE'): 'related_to'
        }
        
        # Process each sentence
        for sent in doc.sents:
            entities = list(sent.ents)
            
            # Add nodes
            for ent in entities:
                if not G.has_node(ent.text):
                    G.add_node(ent.text, type=ent.label_)
            
            # Add edges based on semantic rules
            for ent1 in entities:
                for ent2 in entities:
                    if ent1 != ent2:
                        rule_key = (ent1.label_, ent2.label_)
                        if rule_key in semantic_rules:
                            G.add_edge(ent1.text, ent2.text, 
                                     relationship=semantic_rules[rule_key])
        
        return G

    def visualize_graph(self, G: nx.Graph, output_path: str = None, 
                       title: str = "Entity Knowledge Graph"):
        """Visualize the knowledge graph with improved layout and styling"""
        plt.figure(figsize=(15, 10))
        
        # Use spring layout with adjusted parameters
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw nodes
        for node, (x, y) in pos.items():
            node_type = G.nodes[node].get('type', 'MISC')
            color = self.entity_colors.get(node_type, '#ddd')
            
            nx.draw_networkx_nodes(G, pos, nodelist=[node], 
                                 node_color=color, 
                                 node_size=2000,
                                 alpha=0.7)
        
        # Draw edges with varying width based on weight
        edge_weights = [G[u][v].get('weight', 1) for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_weights,
                             edge_color='gray', alpha=0.5)
        
        # Add labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=color, label=entity_type,
                                    markersize=10)
                         for entity_type, color in self.entity_colors.items()]
        plt.legend(handles=legend_elements, loc='upper left', 
                  bbox_to_anchor=(1, 1))
        
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.show() 