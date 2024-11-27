from typing import Union
import spacy.language
from spacy.tokens import Doc, Token
from profanity_filter.spacy_component import SpacyProfanityFilterComponent
from profanity_filter.types_ import Language


def parse(nlp: spacy.language.Language,
          text: str, 
          language: Language = None,
          use_profanity_filter: bool = False) -> Union[Doc, Token]:
    """Parse text using spaCy pipeline"""
    # In spaCy 3.0, component_cfg is replaced with config
    config = {}
    if use_profanity_filter:
        config["profanity_filter"] = {
            'language': language,
        }
    
    # Create the doc first
    doc = nlp.make_doc(text)
    
    # Apply the pipeline manually, skipping profanity filter if requested
    for name, pipe in nlp.pipeline:
        if not use_profanity_filter and name == "profanity_filter":
            continue
        if name in config:
            doc = pipe(doc, **config[name])
        else:
            doc = pipe(doc)
            
    return doc


def make_token(nlp: spacy.language.Language, word: Union[str, Token]) -> Token:
    """Create a single token from text or return existing token"""
    if hasattr(word, 'text'):
        return word
        
    # Create doc
    doc = nlp.make_doc(word)
    
    # Merge all tokens into one if needed
    if len(doc) > 1:
        with doc.retokenize() as retokenizer:
            attrs = {}  # Preserve token attributes if needed
            retokenizer.merge(doc[:], attrs=attrs)
    
    return doc[0]
