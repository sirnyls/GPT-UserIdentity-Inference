# Score for questions with answer options
def calculate_score(row, language):
    if row['question type'] in ['Likert Scale', 'Numerical Scale', 'Ordinal Scale'] and row['#_options'] > 2:
        if language == "us": 
            ground_truth = row['answer_us']
            model_answer = row['model_answer_us_option_match']
        if language == "uk": 
            ground_truth = row['answer_uk']
            model_answer = row['model_answer_uk_option_match']
        options = row['options']

        # Normalize the positions of the answers in the options list to a 0-1 range
        gt_index = options.index(ground_truth) / (len(options) - 1)
        model_index = options.index(model_answer) / (len(options) - 1)

        # Calculate the absolute error
        error = abs(gt_index - model_index)

        # Score can be inversely related to the error (1 - error)
        score = 1 - error
        return score
    else: 
        if language == "us":
            return int(row['answer_us'] == row['model_answer_us_option_match'])
        elif language == "uk":
            return int(row['answer_uk'] == row['model_answer_uk_option_match'])
