from rasa.nlu.utils.spacy_utils import SpacyNLP as RasaNLUSpacyNLP
from rasa.nlu.config import override_defaults

from .. import spacy_nlp_languages

class SpacyNLP(RasaNLUSpacyNLP):
    name = "optimized_spacy_nlp_with_labels"

    @classmethod
    def load(
        cls, meta, model_dir=None, model_metadata=None, cached_component=None, **kwargs
    ):
        if cached_component:
            return cached_component

        model_name = meta.get("model")

        print('Starting Language Model - Load {}'.format(model_name))

        nlp = spacy_nlp_languages.get(model_name)
        cls.ensure_proper_language_model(nlp)
        return cls(meta, nlp)

    @classmethod
    def create(cls, component_config, config):
        component_config = override_defaults(cls.defaults, component_config)

        spacy_model_name = component_config.get("model")

        # if no model is specified, we fall back to the language string
        if not spacy_model_name:
            component_config["model"] = config.language

        print('Starting Language Model - Create {}'.format(config.language))

        nlp = spacy_nlp_languages.get(config.language)

        cls.ensure_proper_language_model(nlp)
        return cls(component_config, nlp)

    def train(self, training_data, config, **kwargs):
        for example in training_data.training_examples:
            example.set("spacy_doc", self.doc_for_text(example.text))
        if training_data.label_training_examples:
            for example in training_data.label_training_examples:
                example.set("spacy_doc", self.doc_for_text(example.text))
