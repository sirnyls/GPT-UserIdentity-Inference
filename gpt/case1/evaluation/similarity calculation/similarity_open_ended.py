from bert_score import BERTScorer
import pandas as pd
import re
import contractions

scorer = BERTScorer(model_type='microsoft/deberta-xlarge-mnli', num_layers=40)
#scorer = BERTScorer(model_type='microsoft/deberta-v2-xxlarge-mnli', num_layers=22)


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = contractions.fix(text)
    return text


def keyword_matching(short_sentence, long_sentence):
    # Function to remove punctuation
    def remove_punctuation(text):
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)

    short_words = set(remove_punctuation(short_sentence).lower().split())
    long_words = set(remove_punctuation(long_sentence).lower().split())
    return all(word in long_words for word in short_words)

def calculate_similarity(data, language):
    similarity_scores = []

    for index, row in data.iterrows():
        if pd.isna(row['options']):
            if language == 'us': 
                sentence_1 = row['answer_us']
                sentence_2 = row['model_answer_us']
            elif language == 'uk': 
                sentence_1 = row['answer_uk']
                sentence_2 = row['model_answer_uk']
            elif language == 'ground_truths': 
                sentence_1 = row['answer_us']
                sentence_2 = row['answer_uk']
            # Apply keyword matching for short sentences
            if len(sentence_1.split()) <= 2 or len(sentence_2.split()) <= 2:
                if keyword_matching(sentence_1, sentence_2) or keyword_matching(sentence_2, sentence_1):
                    similarity_scores.append(1)
                else:
                    # Apply BERTScorer if keyword matching fails
                    if language == 'us': 
                        sentence_1 = row['answer_us']
                        sentence_2 = row['model_answer_us']
                    elif language == 'uk': 
                        sentence_1 = row['answer_uk']
                        sentence_2 = row['model_answer_uk']
                    elif language == 'ground_truths': 
                        sentence_1 = row['answer_us']
                        sentence_2 = row['answer_uk']
                    P, R, F = scorer.score([sentence_1], [sentence_2])
                    similarity_scores.append(P.item())
            else:
                # Apply BERTScorer for longer sentences
                if language == 'us': 
                    sentence_1 = row['answer_us']
                    sentence_2 = row['model_answer_us']
                elif language == 'uk': 
                    sentence_1 = row['answer_uk']
                    sentence_2 = row['model_answer_uk']
                elif language == 'ground_truths': 
                    sentence_1 = row['answer_us']
                    sentence_2 = row['answer_uk']
                P, R, F = scorer.score([sentence_1], [sentence_2])
                similarity_scores.append(P.item())
        else:
            similarity_scores.append(None)
        

    if language == 'us': 
        data['BERTScore_us_new'] = similarity_scores
    elif language == 'uk':
        data['BERTScore_uk_new'] = similarity_scores
    elif language == 'ground_truths':
        data['BERTScore_ground_truths'] = similarity_scores

