import numpy as np

''' This code is modified based on the evaluation function from the Sentence-Transformers library: 
https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/evaluation/BinaryClassificationEvaluator.py,  and the Sentence-Transformers library is under Apache License 2.0'''



def find_best_acc_and_threshold(scores, labels,):
        high_score_more_similar=True
        assert len(scores) == len(labels)
        rows = list(zip(scores, labels))

        rows = sorted(rows, key=lambda x: x[0], reverse=high_score_more_similar)

        max_acc = 0
        best_threshold = -1
        positive_so_far = 0
        remaining_negatives = sum(labels == 0)

        for i in range(len(rows)-1):
            score, label = rows[i]
            if label == 1:
                positive_so_far += 1
            else:
                remaining_negatives -= 1

            acc = (positive_so_far + remaining_negatives) / len(labels)
            if acc > max_acc:
                max_acc = acc
                best_threshold = (rows[i][0] + rows[i+1][0]) / 2

        return max_acc, best_threshold

def find_best_f1_and_threshold(scores, labels):
        high_score_more_similar=True
        assert len(scores) == len(labels)

        scores = np.asarray(scores)
        labels = np.asarray(labels)

        rows = list(zip(scores, labels))
        rows = sorted(rows, key=lambda x: x[0], reverse=high_score_more_similar)

        best_f1 = best_precision = best_recall = 0
        threshold = 0
        nextract = 0
        ncorrect = 0
        total_num_duplicates = sum(labels)

        for i in range(len(rows)-1):
            score, label = rows[i]
            nextract += 1
            if label == 1:
                ncorrect += 1

            if ncorrect > 0:
                precision = ncorrect / nextract
                recall = ncorrect / total_num_duplicates
                f1 = 2 * precision * recall / (precision + recall)
                if f1 > best_f1:
                    best_f1 = f1
                    best_precision = precision
                    best_recall = recall
                    threshold = (rows[i][0] + rows[i + 1][0]) / 2

        return best_f1, best_precision, best_recall, threshold

# def find_auc_ranking(scores, labels, high_score_more_similar: bool):
#         assert len(scores) == len(labels)
#         scores = np.asarray(scores)
#         labels = np.asarray(labels)
#         rows = list(zip(scores, labels))
#         rows = sorted(rows, key=lambda x: x[0], reverse=high_score_more_similar)

#         score_list=[]
#         label_list=[]
#         for i in range(len(rows)-1):
#             score, label = rows[i]
#             score_list=score_list+[score]
#             label_list=label_list+[label]

#         scores_arr = np.array(score_list)
#         label_arr = np.array(label_list)
    
#         precision, recall, _ = precision_recall_curve(scores_arr , label_arr)
#         auc_score = auc(recall, precision)
#         return auc_score