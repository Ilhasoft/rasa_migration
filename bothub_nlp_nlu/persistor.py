import base64
from tempfile import NamedTemporaryFile

from rasa.nlu.persistor import Persistor


class BothubPersistor(Persistor):
    def __init__(self, update=None, connection=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update = update
        self.connection = connection

    def _persist_tar(self, filekey, tarname):
        with open(tarname, 'rb') as tar_file:
            data = tar_file.read()
            # open('/Users/danielyohan/PycharmProjects/rasa_test_train_old/teste.tar.gz', 'wb').write(data)
            print('salvando repo update {}'.format(self.update))
            init_cursor = self.connection.cursor()
            init_cursor.execute('update common_repositoryupdate set bot_data = %s where id = %s;', (base64.b64encode(data).decode('utf-8'), self.update))
            self.connection.commit()
            print('Save Training')
            # self.update.save_training(data)

    def retrieve(self, model_name, target_path):
        print('chamou aqui')
        tar_name = self._tar_name(model_name)

        # tar_data = self.update.get_bot_data()
        tar_data = open('/Users/danielyohan/PycharmProjects/rasa_test_train_old/teste.tar.gz', 'rb').read()
        tar_file = NamedTemporaryFile(suffix=tar_name, delete=False)
        tar_file.write(tar_data)
        tar_file.close()

        self._decompress(tar_file.name, target_path)
