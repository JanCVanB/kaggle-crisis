from csv import reader
from csv import writer


def read(filepath):
    predictions = []
    with open(filepath, 'rU') as datafile:
        datareader = reader(datafile)
        next(datareader)
        for row in datareader:
            predictions.append(row[1])
    return predictions

rf_tf_idf = read('predictions/predictions_rf_tf_idf.txt')
rf_wc = read('predictions/predictions_rf_wc.txt')
ada_wc = read('predictions/predictions_adaboost.txt')

predictors = (rf_tf_idf, rf_wc, ada_wc)
predictions = []
for i in range(len(rf_tf_idf)):
    vote = sum([int(predictor[i]) for predictor in predictors])
    if vote >= 2:
        predictions.append(1)
    else:
        predictions.append(0)

with open('predictions/predictions_vote.txt', 'wb') as votefile:
    votewriter = writer(votefile)
    votewriter.writerow(['Id', 'Prediction'])
    for i, prediction in enumerate(predictions):
        votewriter.writerow([i+1, prediction])
