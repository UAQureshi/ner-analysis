from typing import List

class EntityAwareChunker:
    def __init__(self, nlp):
        self.nlp = nlp

    def chunk_document(self, text: str, max_chunk_size: int = 512) -> List[str]:
        """Chunk document while preserving entity boundaries"""
        doc = self.nlp(text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sent in doc.sents:
            # Check if adding this sentence exceeds chunk size
            if current_size + len(sent) > max_chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0
            
            current_chunk.append(sent.text)
            current_size += len(sent)
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

    def get_chunk_entities(self, chunk: str) -> List[tuple]:
        """Extract entities from a chunk"""
        doc = self.nlp(chunk)
        return [(ent.text, ent.label_) for ent in doc.ents] 