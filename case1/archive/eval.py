from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from scipy.stats import pearsonr, weightedtau
import pandas as pd

# Base Scorer class
class Scorer:
    def score(self, ground_truth, model_answer):
        raise NotImplementedError("Subclasses should implement this method")

# Binary question scorer (accuracy as metric)
class BinaryScorer(Scorer):
    def score(self, ground_truth, model_answer):
        lb = LabelBinarizer()
        ground_truth_encoded = lb.fit_transform([ground_truth])[0]
        model_answer_encoded = lb.transform([model_answer])[0]
        return accuracy_score(ground_truth_encoded, model_answer_encoded)

# Multiple choice question scorer
class MultipleChoiceScorer(Scorer):
    def score(self, ground_truth, model_answer):
        return int(ground_truth == model_answer)

# Numerical scale question scorer
class NumericalScaleScorer(Scorer):
    def score(self, ground_truth, model_answer):
        return 1 / (1 + mean_squared_error([float(ground_truth)], [float(model_answer)], squared=False))

# Free text question scorer
class FreeTextScorer(Scorer):
    def score(self, ground_truth, model_answer):
        # This simplistic implementation just checks for an exact match
        # In practice, a more complex text similarity measure would be used
        return float(ground_truth.strip().lower() == model_answer.strip().lower())

# Ordinal scale question scorer
class OrdinalScaleScorer(Scorer):
    def __init__(self, scale):
        self.scale = scale

    def score(self, ground_truth, model_answer):
        truth_index = self.scale.index(ground_truth)
        pred_index = self.scale.index(model_answer)
        return 1 - (abs(truth_index - pred_index) / (len(self.scale) - 1))

# Likert scale question scorer
class LikertScaleScorer(Scorer):
    def score(self, ground_truth, model_answer):
        # Assuming ground_truth and model_answer are integers representing Likert scale points
        return weightedtau([int(ground_truth)], [int(model_answer)])[0]

# DatasetEvaluator uses the appropriate Scorer subclass based on the question type
class DatasetEvaluator:
    def __init__(self, data, generative_model):
        self.data = data
        self.model = generative_model
        self.dataframe = pd.read_csv(data, sep=';')
        self.scorers = {
            'binary': BinaryScorer(),
            'multiple_choice': MultipleChoiceScorer(),
            'numerical_scale': NumericalScaleScorer(),
            'free_text': FreeTextScorer(),
            'ordinal_scale': OrdinalScaleScorer(['poor', 'below_average', 'average', 'good', 'excellent']),
            'likert_scale': LikertScaleScorer(),
        }

    def generate_model_answers_and_scores(self):
        self.dataframe['model_answer'] = self.dataframe.apply(
            lambda row: self.model.generate_answer(row['context']), axis=1)
        self.dataframe['score'] = self.dataframe.apply(
            lambda row: self.scorers[row['type']].score(row['ground_truth_answer'], row['model_answer']), axis=1)

    def save_results(self):
        self.dataframe.to_csv(self.csv_file, index=False)

# Dummy generative model class for the example
class GenerativeModel:
    def generate_answer(self, context):
        return "placeholder_answer"

## Example evaluation

data = 'data_merged.csv' 

generative_model = GenerativeModel()
evaluator = DatasetEvaluator(data, generative_model)

evaluator.generate_model_answers_and_scores()
evaluator.save_results()
