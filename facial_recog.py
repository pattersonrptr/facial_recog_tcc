import pickle

from keras import backend as K

K.set_image_data_format('channels_first')

from fr_utils import *
from inception_blocks_v2 import *

np.set_printoptions(threshold=np.nan)


class FaceRecognizer:

    def __init__(self):

        self.FRmodel = faceRecoModel(input_shape=(3, 96, 96))

        self.load_weights()

        # self.database = {
        #     "patterson": img_to_encoding("images/patterson.jpg", self.FRmodel),
        # }

        self.path_database = os.path.join(os.getcwd(), 'base/database.dat')

        # with open(self.path_database, 'wb') as f:
        #     pickle.dump(self.database, f, pickle.HIGHEST_PROTOCOL)

        self.database = self.load_img_data()

        print(self.database)

    def load_weights(self):

        print('Carregando os pesos...')
        self.FRmodel.compile(optimizer='adam', loss=self.triplet_loss, metrics=['accuracy'])
        load_weights_from_FaceNet(self.FRmodel)

    def insert_new_person(self, name, img):

        # img_path = os.path.join('images', img)

        self.database[name] = img_to_encoding(img, self.FRmodel)

        with open(self.path_database, 'wb') as f:
            pickle.dump(self.database, f, pickle.HIGHEST_PROTOCOL)

    def load_img_data(self):

        if os.path.exists(self.path_database):
            print('>>> carregando banco')
            with open(self.path_database, 'rb') as f:
                return pickle.load(f)
        else:
            return {}

    @staticmethod
    def triplet_loss(y_true, y_pred, alpha=0.2):
        """
        Implementação da função de erro tripla.

        Argumentos:
        y_true -- rótulos true (verdadeiros), necessários quando se define uma perda em Keras, não é necessário nesta função.
        y_pred -- lista Python contendo três objetos:
                âncora -- as codificações para uma imagem de âncora, com as dimensões (None, 128)
                positiva -- as codificações para imagens positivas, com as dimensões (None, 128)
                negativa -- as codificações para imagens negativas, com as dimensões (None, 128)

        Returns:
        loss -- número real, valor de perda (erro)
        """

        anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

        # Computa a distância entre a âncora e a positiva
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)

        # Computa a distância entre a âncora e a positiva
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)

        # subtrai as duas distâncias anteriores e adiciona o alpha
        basic_loss = pos_dist - neg_dist + alpha

        # Pega o máximo entre basic_loss e 0.0
        loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))

        return loss

    def verify(self, image_path, identity):

        # Computa a codificação da imagem
        encoding = img_to_encoding(image_path, self.FRmodel)

        # Computa a distância entre a imagem da câmera e a imagem da base
        dist = np.linalg.norm(encoding - self.database[identity])

        # Libera o acesso se a distância for menor que 0.7, do contrário nega o acesso.
        if dist < 0.7:
            print("Olá " + str(identity) + ", bem vindo!")
            door_open = True
        else:
            print("Você não é " + str(identity) + ", acesso negado.")
            door_open = False

        return dist, door_open

    def who_is_it(self, image_path):

        # Codifica a imagem capturada
        encoding = img_to_encoding(image_path, self.FRmodel)

        # Inicializa a menor distância "min_dist" com um valor muito grande, 100.
        min_dist = 100

        access = False
        identity = ''

        # Encontra a imagem com a codificação mais próxima da imagem capturada
        # Itera sobre o dicionário da base de dados obtendo as chaves (names) e as codificações (db_enc).
        for (name, db_enc) in self.database.items():

            # Computa a distância entre a codificação da imagem capturada e a codificação acorrente da base de dados
            dist = np.linalg.norm(encoding - db_enc)

            # Se a distância for menor que a distância mínima, então passa a ser a nova distância mínima,
            # e a identidade passa a ser o nome corrente.
            if dist < min_dist:
                min_dist = dist
                identity = name

        if min_dist > 0.7:
            identity = 'Desconhecido'
            min_dist = str(min_dist)
            access = False

            print("Não existe na base de dados.")

        else:
            identity = str(identity)
            min_dist = str(min_dist)
            access = True

            print("Olá " + identity + ", a distância é " + min_dist)

        identity = str(identity)
        min_dist = str(min_dist)

        return access, min_dist, identity
