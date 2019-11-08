import base64
import json
import os
import re
import tarfile
from tempfile import NamedTemporaryFile, mkdtemp

import psycopg2

from core.train import train_update


class Migration:
    def __init__(self):
        self.ner_spacy = False
        self.similarity_type = 'cosine'
        self.con = psycopg2.connect(host=str(os.environ.get('HOST')), database=str(os.environ.get('DB')), user=str(os.environ.get('USER')), password=str(os.environ.get('PASSWORD')))
        self.get_repository_update()
        self.con.close()

    def get_repository_update(self):
        init_cursor = self.con.cursor()
        init_cursor.execute('''
            SELECT
                   m.id as update,
                   m.algorithm as algorithm,
                   m.bot_data as bot_data,
                   m.repository_id as repository_id
            FROM (
                SELECT repository_id, max(created_at) as date, language
                FROM common_repositoryupdate
                where trained_at is not null
                GROUP BY repository_id, language
            ) as t
                INNER JOIN common_repositoryupdate as m ON m.repository_id = t.repository_id AND t.date = m.created_at
                INNER JOIN common_repository cr on m.repository_id = cr.uuid;''')

        repositorios = init_cursor.fetchall()
        for rec in repositorios:
            self.start(rec[0], rec[1], rec[2], rec[3])

    def start(self, update_id, algorithm, bot_data, repository_uuid):
        model_directory = mkdtemp()
        print(model_directory)
        open(str(model_directory)+'/{}.tar.gz'.format(str(update_id)), 'wb').write(base64.b64decode(bot_data))
        tar_data = open(str(model_directory)+'/{}.tar.gz'.format(str(update_id)), 'rb').read()
        tar_file = NamedTemporaryFile(suffix='{}.tar.gz'.format(str(update_id)), delete=False)
        tar_file.write(tar_data)
        tar_file.close()

        with tarfile.open(tar_file.name, "r:gz") as tar:
            tar.extractall(model_directory)

        files_directory = [
            f for f in os.listdir(model_directory) if re.search(r'^(metadata|training_data)+\.[json]{4}$', f)
        ]

        if len(files_directory) < 2:
            print(
                'Update_ID {} não tem arquivos suficientes para processar o treinamento. Repository: {}'.format(
                    str(update_id),
                    str(repository_uuid)
                )
            )
            return

        training_data = json.loads(open(str(model_directory)+'/training_data.json', 'r').read())
        metadata = json.loads(open(str(model_directory)+'/metadata.json', 'r').read())

        for pipeline in metadata.get('pipeline'):
            if pipeline.get('name') == 'ner_spacy':
                self.ner_spacy = True
            if pipeline.get('name') == 'intent_classifier_tensorflow_embedding':
                self.similarity_type = pipeline.get('similarity_type')

        rasa_nlu_data = training_data.get('rasa_nlu_data')

        train_update(
            update=update_id,
            examples_data=rasa_nlu_data.get('common_examples'),
            label_examples_data=rasa_nlu_data.get('label_examples'),
            algorithm=algorithm,
            ner_spacy=self.ner_spacy,
            similarity_type=self.similarity_type,
            language=metadata.get('language'),
            connection=self.con
        )


if __name__ == '__main__':
    Migration()
