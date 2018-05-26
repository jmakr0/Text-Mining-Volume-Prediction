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
    model.fit([preprocessor.data['headlines'],
               preprocessor.data['hours'],
               preprocessor.data['minutes'],
               preprocessor.data['day_of_weeks'],
               preprocessor.data['day_of_years']],
              [preprocessor.data['is_top_submission'],
               preprocessor.data['is_top_submission']],
              batch_size=64,
              epochs=20,
              validation_split=0.2,
              callbacks=[csv_logger])


if __name__ == '__main__':
    main()
