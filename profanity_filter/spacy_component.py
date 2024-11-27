from contextlib import suppress
from typing import Union, Optional, Generator, List
import spacy
from spacy.language import Language
from more_itertools import partitions
from spacy.tokens import Doc, Span, Token
from profanity_filter import spacy_utlis
from profanity_filter.types_ import Language as PFLanguage


class SpacyProfanityFilterComponent:
    def __init__(
        self,
        profanity_filter: 'ProfanityFilter',
        nlp: Language,
        name: str,
        language: Optional[PFLanguage] = None,
        stop_on_first_profane_word: bool = False
    ):
        self._language = language
        self._nlp = nlp  # Used only for tokenization
        self._profanity_filter = profanity_filter
        self._stop_on_first_profane_word = stop_on_first_profane_word
        self.name = name

    def __call__(
        self,
        doc: Doc,
        language: Optional[PFLanguage] = None,
        stop_on_first_profane_word: Optional[bool] = None
    ) -> Doc:
        self.register_extensions(exist_ok=True)
        
        if language is None:
            language = self._language
        if stop_on_first_profane_word is None:
            stop_on_first_profane_word = self._stop_on_first_profane_word

        i = 0
        while i < len(doc):
            j = i + 1
            while (j < len(doc) 
                   and not doc[j - 1].whitespace_ 
                   and not doc[j - 1].is_space 
                   and not doc[j - 1].is_punct
                   and not doc[j].is_space 
                   and not doc[j].is_punct):
                j += 1
            span = self._censor_spaceless_span(doc[i:j], language=language)
            if stop_on_first_profane_word and span._.is_profane:
                break
            i += len(span)
        return doc

    @staticmethod
    def register_extensions(exist_ok: bool = False) -> None:
        def do() -> None:
            if not Token.has_extension("censored"):
                Token.set_extension("censored", default=None)
            if not Token.has_extension("is_profane"):
                Token.set_extension("is_profane", getter=SpacyProfanityFilterComponent.token_is_profane)
            if not Token.has_extension("original_profane_word"):
                Token.set_extension("original_profane_word", default=None)
            if not Span.has_extension("is_profane"):
                Span.set_extension("is_profane", getter=SpacyProfanityFilterComponent.tokens_are_profane)
            if not Doc.has_extension("is_profane"):
                Doc.set_extension("is_profane", getter=SpacyProfanityFilterComponent.tokens_are_profane)

        if exist_ok:
            with suppress(ValueError):
                do()
        else:
            do()

    @staticmethod
    def token_is_profane(token: Token) -> bool:
        return token._.censored != token.text

    @staticmethod
    def tokens_are_profane(tokens: Union[Doc, Span]) -> bool:
        return any(token._.is_profane for token in tokens)

    def _span_partitions(self, span: Span) -> Generator[List[Token], None, None]:
        if len(span) == 1:
            yield [span[0]]
            return
        
        for partition in partitions(span):
            yield [spacy_utlis.make_token(nlp=self._nlp, word=''.join(token.text for token in element)) 
                  for element in partition]

    def _censor_spaceless_span(self, span: Span, language: PFLanguage) -> Span:
        token = spacy_utlis.make_token(
            nlp=self._nlp,
            word=span.text if len(span) > 1 else span[0].text
        )
        
        censored_word = self._profanity_filter.censor_word(word=token, language=language)
        
        if censored_word.is_profane:
            with span.doc.retokenize() as retokenizer:
                attrs = {
                    "LEMMA": token.lemma_,
                    "_": {"censored": censored_word.censored, 
                         "original_profane_word": censored_word.original_profane_word}
                }
                retokenizer.merge(span, attrs=attrs)
        else:
            for token in span:
                token._.censored = token.text
                
        return span
