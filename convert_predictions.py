from csv import reader
from csv import writer


with open('results/test.out') as rf_result_file:
    rf_result_reader = reader(rf_result_file)
    next(rf_result_reader)
    with open('results/predictions_svm.txt', 'wb') as rf_predictions_file:
        rf_predictions_writer = writer(rf_predictions_file)
        rf_predictions_writer.writerow(['Id', 'Prediction'])
        for row in rf_result_reader:
            rf_predictions_writer.writerow([entry.strip('"') for entry in row[0].split()])
