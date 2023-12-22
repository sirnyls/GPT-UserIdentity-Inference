import textstat
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import language_tool_python 
import statistics

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

def analyze_essay(essay):
    """
    Analyze various aspects of an essay for linguistic sophistication.

    :param essay: A string containing the essay text.
    :return: A dictionary with various metrics and an overall quality score.
    """
    
    #lt = language_tool_python.LanguageToolPublicAPI('en-US')

    # Readability and complexity metrics
    flesch_reading_ease = textstat.flesch_reading_ease(essay)
    smog_index = textstat.smog_index(essay)
    flesch_kincaid_grade = textstat.flesch_kincaid_grade(essay)

    # Tokenize essay into words and sentences
    words = word_tokenize(essay)
    sentences = sent_tokenize(essay)
    unique_words = len(set(words))

    # Lexical diversity
    lexical_diversity = len(set(words)) / len(words) if words else 0

    # Sentence length variability
    sentence_lengths = [len(word_tokenize(sentence)) for sentence in sentences]
    sentence_length_variability = statistics.stdev(sentence_lengths) if len(sentence_lengths) > 1 else 0

    # Grammatical mistakes
    #matches = lt.check(essay)
    #grammar_errors = len(matches) 
    grammar_errors = 0

    # Complexity of sentence structures (using POS tagging)
    pos_tags = [pos for word, pos in nltk.pos_tag(words)]
    complex_structures = pos_tags.count('JJ') + pos_tags.count('NNP') + pos_tags.count('VBD')

    # Calculate overall quality score (example formula, can be customized)
    overall_quality = (
        (flesch_reading_ease + 100 - smog_index + 100 - flesch_kincaid_grade) +
        (lexical_diversity * 100 + unique_words * 0.5) - 
        (grammar_errors + sentence_length_variability - complex_structures)
    ) / 8
    return   {
        "flesch_reading_ease": flesch_reading_ease,
        "smog_index": smog_index,
        "flesch_kincaid_grade": flesch_kincaid_grade,
        "unique_words": unique_words,
        "lexical_diversity": lexical_diversity,
        "sentence_length_variability": sentence_length_variability,
        "grammar_errors": grammar_errors,
        "complex_structures": complex_structures,
        "overall_quality": overall_quality
    }
 #overall_quality

# Load essays
#essay1 = data.iloc[7, 5]
#essay2 = data.iloc[7, 6]

#linguistic sophistication
#results1 = analyze_essay(essay1)
#results2 = analyze_essay(essay2)
#print(results1)
#print(results2)

