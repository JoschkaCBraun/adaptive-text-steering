"""
Validate inputs for the topic vector training samples.
"""


def validate_topic_representation_type(topic_representation_type: str) -> None:
    '''Validate the topic representation type.
    '''
    if topic_representation_type not in ['topic_words', 'topic_phrases', 'topic_descriptions', 'topic_summaries']:
        raise ValueError(f"Invalid topic representation type: {topic_representation_type}")
    
def validate_pairing_type(pairing_type: str) -> None:
    '''Validate the pairing type.
    '''
    if pairing_type not in ['random_topic_representation', 'random_string']:
        raise ValueError(f"Invalid pairing type: {pairing_type}")
    
def validate_language(language: str) -> None:
    '''Validate the language.
    '''
    if language not in ['en', 'fr', 'de']:
        raise ValueError(f"Invalid language: {language}")
