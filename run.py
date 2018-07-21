from argparse import ArgumentParser, RawTextHelpFormatter

from src.models.model_1 import train as train_model_1
from src.models.model_2 import train as train_model_2
from src.models.model_3 import train as train_model_3
from src.models.model_4 import train as train_model_4
from src.models.model_5 import train as train_model_5
from src.models.model_6 import train as train_model_6
from src.models.model_7 import train as train_model_7

from src.models.model_23 import train as train_model_23
from src.models.model_24 import train as train_model_24
from src.models.model_25 import train as train_model_25
from src.models.model_26 import train as train_model_26
from src.models.model_27 import train as train_model_27
from src.models.model_34 import train as train_model_34
from src.models.model_234 import train as train_model_234

from src.evaluators.correlation_evaluator import calculate_correlations
from src.encoder.competitive_score import CompetitiveScore
from src.guardian.downloader import download as guardian_api_download
from src.data_handler.guardian_csv import GuardianCsvData


class Action:
    def __init__(self, action, description):
        self.action = action
        self.description = description


class TrainAll:
    def __init__(self, train_actions):
        self.train_actions = train_actions

    def __call__(self):
        for action in train_actions.values():
            if action.action != self:
                action.action()


def doc2vec_upload():
    cs = CompetitiveScore()
    cs.save_doc2vec()


def guardian_csv_import():
    guardian_csv = GuardianCsvData()
    guardian_csv.import_article()
    guardian_csv.import_authors()
    guardian_csv.import_comments()


if __name__ == '__main__':
    train_actions = {}
    train_actions['0'] = Action(TrainAll(train_actions), 'Train all models.')
    train_actions['1'] = Action(train_model_1, 'Train headline dense model.')
    train_actions['2'] = Action(train_model_2, 'Train headline convolution model.')
    train_actions['3'] = Action(train_model_3, 'Train article LSTM model.')
    train_actions['4'] = Action(train_model_4, 'Train category model.')
    train_actions['5'] = Action(train_model_5, 'Train time model.')
    train_actions['6'] = Action(train_model_6, 'Train headline and article length model.')
    train_actions['7'] = Action(train_model_7, 'Train competitive score model.')
    train_actions['23'] = Action(train_model_23, 'Train headline convolution and article LSTM model.')
    train_actions['24'] = Action(train_model_24, 'Train headline convolution and category model.')
    train_actions['25'] = Action(train_model_25, 'Train headline convolution and time model.')
    train_actions['26'] = Action(train_model_26, 'Train headline convolution and headline and article length model.')
    train_actions['27'] = Action(train_model_27, 'Train headline convolution and competitive score model.')
    train_actions['34'] = Action(train_model_34, 'Train article LSTM and category model.')
    train_actions['234'] = Action(train_model_234, 'Train headline convolution and article LSTM and category model.')

    general_actions = {
        'calc_correlation': Action(calculate_correlations, 'Calculate base model correlations.'),
        'doc2vec_upload': Action(doc2vec_upload, 'Calculate and upload all doc2vec vectors for article text corpus.'),
        'download': Action(guardian_api_download, 'Download articles from the guardian API.'),
        'csv_import': Action(guardian_csv_import, 'Import the guardian dataset from the CSVs into the database.')
    }

    train_help = '\n'.join([key + ': ' + action.description for key, action in train_actions.items()])
    general_help = '\n'.join([key + ': ' + action.description for key, action in general_actions.items()])

    arg_parse = ArgumentParser(formatter_class=RawTextHelpFormatter)
    arg_parse.add_argument('--train', choices=train_actions.keys(), help=train_help)
    arg_parse.add_argument('--action', choices=general_actions.keys(), help=general_help)
    arguments = arg_parse.parse_args()

    if arguments.train:
        train_actions[arguments.train].action()

    if arguments.action:
        general_actions[arguments.action].action()
