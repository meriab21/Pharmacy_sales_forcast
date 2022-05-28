from pickle import load
from scripts.data_loader import load_df_from_csv
from scripts.logger_creator import CreateLogger
from scripts.ML_modelling_utils import calculate_metrics
from datetime import datetime

try:
    # Create Logger
    logger = CreateLogger('CML-Updater', handlers=1)
    logger = logger.get_default_logger()
    # Importing Model
    model_name = '01-08-2021-21-23-15-74.17%.pkl'
    model_dir = './models/'+model_name
    with open(model_dir , 'rb') as handle:
        model = load(handle)
        logger.info('Model Loaded Successfully!')
    # Importing data to test with
    test_data = load_df_from_csv('./models/test.csv')
    # Get x_value and y_values
    y_values = test_data['Sales']
    x_values = test_data.drop(['Sales'], axis=1)
    logger.info('Test Data Loaded Successfully!')

    # Score and Store Result in a TextFile
    score = model.score(x_values, y_values)
    metrics = calculate_metrics(y_values, model.predict(x_values))
    rmse = metrics['RMSE Score']
    r_sq = metrics['R2_Squared']
    mae = metrics['MAE Score']

    date = datetime.now()
    date = date.strftime("%A-%B-%Y : %I-%M-%S %p")
    
    with open('./models/results.txt', 'w') as file:
        file.write(f'Date:\n\t{date}\n')
        file.write(f'Model Name:\n\t{model_name}\n')
        file.write('Test Score:\n\t{:2%}\n'.format(score))
        file.write('Metrics:\n')
        file.write(f'RMSE:\n\t{rmse}\n')
        file.write(f'R2Error:\n\t{r_sq}\n')
        file.write(f'MAE:\n\t{mae}\n')
        logger.info('Report Generated Successfully!')

except Exception as e:
    logger.exception("Failed to Update CML")
