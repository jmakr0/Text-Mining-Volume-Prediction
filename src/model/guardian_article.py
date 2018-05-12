from src.model.article import Article
from src.utils.guardian_labels import GuardianLabels
from src.utils.labels import Labels


class GuardianArticle(Article):
    def __init__(self, article_id, article_dict):
        label = Labels

        new_article_dict = {}
        for label in list(GuardianLabels):
            if label.value in article_dict:
                new_article_dict[getattr(Labels, label.name).value] = article_dict[label.value]

        super().__init__(article_id, new_article_dict)
