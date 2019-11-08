import logging

from .utils import UpdateInterpreters
from .utils import SpacyNLPLanguageManager


logger = logging.getLogger('bothub_nlp.bothub_nlp_nlu')

updateInterpreters = UpdateInterpreters()
spacy_nlp_languages = SpacyNLPLanguageManager()
