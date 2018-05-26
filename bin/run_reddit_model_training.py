from keras.callbacks import CSVLogger

from src.data_handler.labels_db import LabelsDb
from src.models.glove import Glove
from src.prediction.reddit_model import RedditModelPreprocessor, RedditModelBuilder


def main():
    dictionary_size = 40000
    max_headline_length = 20

    glove = Glove(dictionary_size)
    glove.load_embedding()
    db = LabelsDb()

    preprocessor = RedditModelPreprocessor(glove, db, 20)
    preprocessor.load_data()

    model_builder = RedditModelBuilder()\
        .set_input('glove', glove)\
        .set_parameter('max_headline_length', max_headline_length)

    model = model_builder()

    csv_logger = CSVLogger('training.csv')
    model.fit([preprocessor.training_data['headlines'],
               preprocessor.training_data['hours'],
               preprocessor.training_data['minutes'],
               preprocessor.training_data['day_of_weeks'],
               preprocessor.training_data['day_of_years']],
              [preprocessor.training_data['is_top_submission'],
               preprocessor.training_data['is_top_submission']],
              batch_size=64,
              epochs=20,
              validation_data=([preprocessor.validation_data['headlines'],
               preprocessor.validation_data['hours'],
               preprocessor.validation_data['minutes'],
               preprocessor.validation_data['day_of_weeks'],
               preprocessor.validation_data['day_of_years']],
              [preprocessor.validation_data['is_top_submission'],
               preprocessor.validation_data['is_top_submission']]),
              callbacks=[csv_logger],
              class_weight=preprocessor.validation_data['class_weights']
              )


if __name__ == '__main__':
    main()
