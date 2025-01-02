import pandas as pd
from typing import List, Tuple, Dict

class NERProcessor:
    def __init__(self, nlp, ner_transformer):
        self.nlp = nlp
        self.ner_transformer = ner_transformer

    def process_with_spacy(self, text: str) -> pd.DataFrame:
        """Process text using spaCy NER"""
        doc = self.nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return pd.DataFrame(entities, columns=['Entity', 'Label']), doc

    def process_with_transformer(self, text: str) -> pd.DataFrame:
        """Process text using transformer-based NER"""
        results = self.ner_transformer(text)
        
        # Filter out subword tokens and duplicates
        filtered_results = []
        seen_words = set()
        
        for result in results:
            if '##' not in result['word'] and result['word'] not in seen_words:
                filtered_results.append(result)
                seen_words.add(result['word'])
        
        return pd.DataFrame(filtered_results)

    def compare_results(self, text: str) -> pd.DataFrame:
        """Compare results from both approaches"""
        # spaCy results
        doc = self.nlp(text)
        spacy_entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        # Transformer results
        transformer_results = self.ner_transformer(text)
        transformer_entities = [(result['word'], result['entity']) 
                              for result in transformer_results]
        
        # Create comparison DataFrame
        comparison = pd.DataFrame({
            'spaCy': pd.Series(dict(spacy_entities)),
            'Transformer': pd.Series(dict(transformer_entities))
        })
        
        return comparison 