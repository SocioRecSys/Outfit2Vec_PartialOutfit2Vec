### Shatha Jaradat
### KTH - Royal Institute of Technology
### 2019

### Some parts of this code was taken from the below link and customised to my needs
### https://github.com/rdevooght/sequence-based-recommendations
### Modified some methods and added some new ones

import math

'''
Evaluation class - to compute evaluation metrics on tests
It is used by adding a series of instances: pairs of goals and predictions
then metrics can be computed on the ensemble of instances:
average precision, average recall, average ncdg, average hit ratio
Returns the set of correct predictions
'''
# Handles Structured Entities and Structured Words
class Evaluator(object):

    def __init__(self, dataset, k, isStructured):
        self.instances = []
        self.dataset = dataset
        self.k = k
        self.isStructured = isStructured

        self.metrics = {
            'recall': self.average_recall,
            'precision': self.average_precision,
            'ndcg': self.average_ndcg,
            'hit_ratio': self.average_hitRatio
        }

    def add_instance(self, goal, predictions):
        self.instances.append([goal, predictions])

    def average_precision(self):
        precision = 0
        for goal, prediction in self.instances:
            lstPrediction = []
            # based on k populate the list
            for i in range(self.k):#range(min(len(prediction),self.k)):
                lstPrediction.append(prediction[i][0])

            lstWordsInGoal = []
            lstWordsInPrediction = []
            for word in goal:
                if self.isStructured:
                    tempList = []
                    for subword in word.split(' '):
                        tempList.append(subword)
                    for subword in tempList:
                        for sub in subword.split('_'):
                            lstWordsInGoal.append(sub)
                else:
                    for subword in word.split(' '):
                        lstWordsInGoal.append(subword)

            for word in lstPrediction:
                if self.isStructured:
                    tempList = []
                    for subword in word.split(' '):
                        tempList.append(subword)
                    for subword in tempList:
                        for sub in subword.split('_'):
                            lstWordsInPrediction.append(sub)
                else:
                    for subword in word.split(' '):
                        lstWordsInPrediction.append(subword)

            precision += float(len([value for value in lstWordsInGoal if value in lstWordsInPrediction])) / min(len(lstWordsInPrediction),self.k)

        return precision / len(self.instances)

    def average_recall(self):
        recall = 0

        for goal, prediction in self.instances:
            lstPrediction = []
            # based on k populate the list
            for i in  range(self.k):#range(min(len(prediction),self.k)):
                lstPrediction.append(prediction[i][0])

            # based on k populate the list
            lstWordsInGoal = []
            lstWordsInPrediction = []
            for word in goal:
                if self.isStructured:
                    tempList = []
                    for subword in word.split(' '):
                        tempList.append(subword)
                    for subword in tempList:
                        for sub in subword.split('_'):
                            lstWordsInGoal.append(sub)

                else:
                    for subword in word.split(' '):
                        lstWordsInGoal.append(subword)

            for word in lstPrediction:
                if self.isStructured:
                    tempList = []
                    for subword in word.split(' '):
                        tempList.append(subword)
                    for subword in tempList:
                        for sub in subword.split('_'):
                            lstWordsInPrediction.append(sub)
                else:
                    for subword in word.split(' '):
                        lstWordsInPrediction.append(subword)

            recall += float(len([value for value in lstWordsInGoal if value in lstWordsInPrediction])) / len(lstWordsInGoal)

        return recall / len(self.instances)

    def average_hitRatio(self):
        hrk_score = 0.0
        #print("hit ratio")
        for goal, prediction in self.instances:
             lstPrediction = []
                # based on k populate the list
             for i in  range(self.k):#range(min(len(prediction),self.k)):
                 lstPrediction.append(prediction[i][0])

                 lstWordsInGoal = []
                 lstWordsInPrediction = []
                 for word in goal:
                    if self.isStructured:
                        tempList = []
                        for subword in word.split(' '):
                            tempList.append(subword)
                        for subword in tempList:
                            for sub in subword.split('_'):
                                lstWordsInGoal.append(sub)

                    else:
                        for subword in word.split(' '):
                            lstWordsInGoal.append(subword)

                 for word in lstPrediction:
                     if self.isStructured:
                        tempList = []
                        for subword in word.split(' '):
                            tempList.append(subword)
                        for subword in tempList:
                            for sub in subword.split('_'):
                                lstWordsInPrediction.append(sub)
                     else:
                        for subword in word.split(' '):
                            lstWordsInPrediction.append(subword)

                 if str(goal[0]) == str(prediction[0][0]):
                    # HR@k
                    hrk_score += 1/float(self.k)
                 elif (len([value for value in lstWordsInGoal if value in lstWordsInPrediction]) > 0):
                    hrk_score += 1/float(self.k)

        return hrk_score / len(self.instances)

    def average_ndcg(self):
            ndcg = 0.
            for goal, prediction in self.instances:
                lstPrediction = []
            # based on k populate the list
            for i in  range(self.k):#range(min(len(prediction),self.k)):
                lstPrediction.append(prediction[i][0])

                lstWordsInGoal = []
                lstWordsInPrediction = []
                for word in goal:
                    if self.isStructured:
                        tempList = []
                        for subword in word.split(' '):
                            tempList.append(subword)
                        for subword in tempList:
                            for sub in subword.split('_'):
                                lstWordsInGoal.append(sub)

                    else:
                        for subword in word.split(' '):
                            lstWordsInGoal.append(subword)

                for word in lstPrediction:
                    if self.isStructured:
                        tempList = []
                        for subword in word.split(' '):
                            tempList.append(subword)
                        for subword in tempList:
                            for sub in subword.split('_'):
                                lstWordsInPrediction.append(sub)
                    else:
                        for subword in word.split(' '):
                            lstWordsInPrediction.append(subword)

                    value = 0.0
                    for i, p in enumerate(lstWordsInPrediction):

                        # check the main categories ?
                        # ndcg for materials, patterns, ...
                        for word in lstWordsInGoal:
                            if word == p:
                                value =  math.log(2) / math.log(i+2)
                        ndcg += value

            return ndcg / len(self.instances)