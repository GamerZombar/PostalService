import json
import os
import sys
import random
import cv2
import warnings

import editdistance
import tensorflow as tf
import numpy as np
from collections import namedtuple
from typing import List, Tuple
from path import Path
from pyaspeller import YandexSpeller
from contextlib import contextmanager
warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')

Sample = namedtuple('Sample', 'gt_text, file_path')
Batch = namedtuple('Batch', 'imgs, gt_texts, batch_size')
img_size = (330, 40)
speller = YandexSpeller()


class Preprocessor:
    def __init__(self,
                 img_size: Tuple[int, int],
                 padding: int = 0,
                 dynamic_width: bool = False,
                 data_augmentation: bool = False,
                 line_mode: bool = False) -> None:
        # dynamic width only supported when no data augmentation happens
        assert not (dynamic_width and data_augmentation)
        # when padding is on, we need dynamic width enabled
        assert not (padding > 0 and not dynamic_width)

        self.img_size = img_size
        self.padding = padding
        self.dynamic_width = dynamic_width
        self.data_augmentation = data_augmentation
        self.line_mode = line_mode


    @staticmethod
    def _truncate_label(text: str, max_text_len: int) -> str:
        """
        Function ctc_loss can't compute loss if it cannot find a mapping between text label and input
        labels. Repeat letters cost double because of the blank symbol needing to be inserted.
        If a too-long label is provided, ctc_loss returns an infinite gradient.
        """
        cost = 0
        for i in range(len(text)):
            if i != 0 and text[i] == text[i - 1]:
                cost += 2
            else:
                cost += 1
            if cost > max_text_len:
                return text[:i]
        return text


    def process_img(self, img: np.ndarray) -> np.ndarray:
        """Resize to target size, apply data augmentation."""

        # data augmentation
        img = img.astype(np.float)
        if self.data_augmentation:
            # photometric data augmentation
            if random.random() < 0.25:
                def rand_odd():
                    return random.randint(1, 3) * 2 + 1
                img = cv2.GaussianBlur(img, (rand_odd(), rand_odd()), 0)
            if random.random() < 0.25:
                img = cv2.dilate(img, np.ones((3, 3)))
            if random.random() < 0.25:
                img = cv2.erode(img, np.ones((3, 3)))

            # geometric data augmentation
            wt, ht = self.img_size
            h, w = img.shape
            f = min(wt / w, ht / h)
            fx = f * np.random.uniform(0.75, 1.05)
            fy = f * np.random.uniform(0.75, 1.05)

            # random position around center
            txc = (wt - w * fx) / 2
            tyc = (ht - h * fy) / 2
            freedom_x = max((wt - fx * w) / 2, 0)
            freedom_y = max((ht - fy * h) / 2, 0)
            tx = txc + np.random.uniform(-freedom_x, freedom_x)
            ty = tyc + np.random.uniform(-freedom_y, freedom_y)

            # map image into target image
            M = np.float32([[fx, 0, tx], [0, fy, ty]])
            target = np.ones(self.img_size[::-1]) * 255
            img = cv2.warpAffine(img, M, dsize=self.img_size, dst=target, borderMode=cv2.BORDER_TRANSPARENT)

            # photometric data augmentation
            if random.random() < 0.5:
                img = img * (0.25 + random.random() * 0.75)
            if random.random() < 0.25:
                img = np.clip(img + (np.random.random(img.shape) - 0.5) * random.randint(1, 25), 0, 255)
            if random.random() < 0.1:
                img = 255 - img

        # no data augmentation
        else:
            if self.dynamic_width:
                ht = self.img_size[1]
                h, w = img.shape
                f = ht / h
                wt = int(f * w + self.padding)
                wt = wt + (4 - wt) % 4
                tx = (wt - w * f) / 2
                ty = 0
            else:
                wt, ht = self.img_size
                h, w = img.shape
                f = min(wt / w, ht / h)
                tx = (wt - w * f) / 2
                ty = (ht - h * f) / 2

            # map image into target image
            M = np.float32([[f, 0, tx], [0, f, ty]])
            target = np.ones([ht, wt]) * 255
            img = cv2.warpAffine(img, M, dsize=(wt, ht), dst=target, borderMode=cv2.BORDER_TRANSPARENT)

        # transpose for TF
        img = cv2.transpose(img)

        # convert to range [-1, 1]
        img = img / 255 - 0.5
        return img


    def process_batch(self, batch: Batch) -> Batch:

        res_imgs = [self.process_img(img) for img in batch.imgs]
        max_text_len = res_imgs[0].shape[0] // 4
        res_gt_texts = [self._truncate_label(gt_text, max_text_len) for gt_text in batch.gt_texts]
        return Batch(res_imgs, res_gt_texts, batch.batch_size)


class DataLoader:
  def __init__(self,
                data_dir: Path,
                batch_size: int,
                data_split: float = 0.9,
                fast: bool = False) -> None:
        """Загрузчик датасета"""

        assert data_dir.exists()

        # self.fast = fast
        # if fast:
        #     self.env = lmdb.open('lmdb', readonly=True)

        self.data_augmentation = False
        self.curr_idx = 0  # порядковый номер в выборке
        self.batch_size = batch_size
        self.samples = []

        filePath = "../images/"

        with open(filePath + 'labels.json', encoding="utf-8") as json_data:
            label_file = json.load(json_data)

        # Log
        print("Loaded", len(label_file), "images")

        # Put sample into list
        for fileName, gtText in label_file.items():
            self.samples.append(Sample(gtText, filePath + fileName))


        # labels_path = Path.joinpath(data_dir, 'labels')
        # file = open(labels_path / 'HTR.csv')
        # reader = csv.reader(file)
        # next(reader)  # пропуск заголовка
        #
        # for row in reader:
        #   gt_text, img_name = row
        #   img_path = Path.joinpath(data_dir, img_name[2:])  # нужно для colab, т.к. пути начинаются с ./
        #   self.samples.append(Sample(gt_text, img_path))
        #
        # file.close()

        # фиксированное деление выборки (по умолчанию на валидацию 10%)
        split_idx = int(data_split * len(self.samples))
        random.seed(6)
        random.shuffle(self.samples)
        self.train_samples = self.samples[:split_idx]
        self.validation_samples = self.samples[split_idx:]

        self.train_texts = [x.gt_text for x in self.train_samples]
        self.validation_texts = [x.gt_text for x in self.validation_samples]

        # start with train set
        self.train_set()


  def train_set(self) -> None:
        """Случайно выбранная подвыборка из тренировочного набора данных"""
        self.data_augmentation = True
        self.curr_idx = 0
        random.seed()
        random.shuffle(self.train_samples)
        self.samples = self.train_samples
        self.curr_set = 'train'


  def validation_set(self) -> None:
        """Вылидационный сет"""
        self.data_augmentation = False
        self.curr_idx = 0
        self.samples = self.validation_samples
        self.curr_set = 'val'


  def get_iterator_info(self) -> Tuple[int, int]:
        """Возвращает номер текущего батча и общее число батчей"""
        if self.curr_set == 'train':
            num_batches = int(np.floor(len(self.samples) / self.batch_size))  # в тренирочной выборке должны быть только полноразмерные батчи
        else:
            num_batches = int(np.ceil(len(self.samples) / self.batch_size))  # в валидационной последний батч может быть меньше
        curr_batch = self.curr_idx // self.batch_size + 1
        return curr_batch, num_batches


  def has_next(self) -> bool:
        """Возвращает флаг, показывающий, хватает ли данных на еще один батч"""
        if self.curr_set == 'train':
            return self.curr_idx + self.batch_size <= len(self.samples)  # в тренирочной выборке должны быть только полноразмерные батчи
        else:
            return self.curr_idx < len(self.samples)  # в валидационной последний батч может быть меньше


  def _get_img(self, i: int) -> np.ndarray:
        img = cv2.imread(self.samples[i].file_path, cv2.IMREAD_GRAYSCALE)

        return img


  def get_batch(self) -> Batch:
        """Получить батч"""
        batch_range = range(self.curr_idx, min(self.curr_idx + self.batch_size, len(self.samples)))

        imgs = [self._get_img(i) for i in batch_range]
        gt_texts = [self.samples[i].gt_text for i in batch_range]

        self.curr_idx += self.batch_size
        return Batch(imgs, gt_texts, len(imgs))


# Disable eager mode
tf.compat.v1.disable_eager_execution()

class DecoderType:
    """CTC decoder types."""
    BestPath = 0
    BeamSearch = 1
    WordBeamSearch = 2


class Model:
    """Minimalistic TF model for HTR."""

    def __init__(self,
                 char_list: List[str],
                 decoder_type: str = DecoderType.BestPath,
                 must_restore: bool = False,
                 dump: bool = False) -> None:
        """Init model: add CNN, RNN and CTC and initialize TF."""
        self.dump = dump
        self.char_list = char_list
        self.decoder_type = decoder_type
        self.must_restore = must_restore
        self.snap_ID = 0

        # Whether to use normalization over a batch or a population
        self.is_train = tf.compat.v1.placeholder(tf.bool, name='is_train')

        # input image batch
        self.input_imgs = tf.compat.v1.placeholder(tf.float32, shape=(None, None, None))

        # setup CNN, RNN and CTC
        self.setup_cnn()
        self.setup_rnn()
        self.setup_ctc()

        # setup optimizer to train NN
        self.batches_trained = 0
        self.update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.optimizer = tf.compat.v1.train.AdamOptimizer().minimize(self.loss)

        # initialize TF

        self.sess, self.saver = self.setup_tf()

    def setup_cnn(self) -> None:
        """Create CNN layers."""
        cnn_in4d = tf.expand_dims(input=self.input_imgs, axis=3)

        # list of parameters for the layers
        kernel_vals = [5, 5, 3, 3, 3]
        feature_vals = [1, 32, 64, 128, 128, 256]
        stride_vals = pool_vals = [(2, 2), (2, 2), (1, 2), (1, 2), (1, 2)]
        num_layers = len(stride_vals)

        # create layers
        pool = cnn_in4d  # input to first CNN layer
        for i in range(num_layers):
            kernel = tf.Variable(
                tf.random.truncated_normal([kernel_vals[i], kernel_vals[i], feature_vals[i], feature_vals[i + 1]],
                                           stddev=0.1))
            conv = tf.nn.conv2d(input=pool, filters=kernel, padding='SAME', strides=(1, 1, 1, 1))
            conv_norm = tf.compat.v1.layers.batch_normalization(conv, training=self.is_train)
            relu = tf.nn.relu(conv_norm)
            pool = tf.nn.max_pool2d(input=relu, ksize=(1, pool_vals[i][0], pool_vals[i][1], 1),
                                    strides=(1, stride_vals[i][0], stride_vals[i][1], 1), padding='VALID')

        self.cnn_out_4d = pool

    def setup_rnn(self) -> None:
        """Create RNN layers."""
        rnn_in3d = tf.squeeze(self.cnn_out_4d, axis=[2])

        # basic cells which is used to build RNN
        num_hidden = 256
        cells = [tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=num_hidden, state_is_tuple=True, reuse=tf.compat.v1.AUTO_REUSE) for _ in
                 range(2)]  # 2 layers

        # stack basic cells
        stacked = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

        # bidirectional RNN
        # BxTxF -> BxTx2H
        (fw, bw), _ = tf.compat.v1.nn.bidirectional_dynamic_rnn(cell_fw=stacked, cell_bw=stacked, inputs=rnn_in3d,
                                                                dtype=rnn_in3d.dtype)

        # BxTxH + BxTxH -> BxTx2H -> BxTx1X2H
        concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)

        # project output to chars (including blank): BxTx1x2H -> BxTx1xC -> BxTxC
        kernel = tf.Variable(tf.random.truncated_normal([1, 1, num_hidden * 2, len(self.char_list) + 1], stddev=0.1))
        self.rnn_out_3d = tf.squeeze(tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'),
                                     axis=[2])

    def setup_ctc(self) -> None:
        """Create CTC loss and decoder."""
        # BxTxC -> TxBxC
        self.ctc_in_3d_tbc = tf.transpose(a=self.rnn_out_3d, perm=[1, 0, 2])
        # ground truth text as sparse tensor
        self.gt_texts = tf.SparseTensor(tf.compat.v1.placeholder(tf.int64, shape=[None, 2]),
                                        tf.compat.v1.placeholder(tf.int32, [None]),
                                        tf.compat.v1.placeholder(tf.int64, [2]))

        # calc loss for batch
        self.seq_len = tf.compat.v1.placeholder(tf.int32, [None])
        self.loss = tf.reduce_mean(
            input_tensor=tf.compat.v1.nn.ctc_loss(labels=self.gt_texts, inputs=self.ctc_in_3d_tbc,
                                                  sequence_length=self.seq_len,
                                                  ctc_merge_repeated=True))

        # calc loss for each element to compute label probability
        self.saved_ctc_input = tf.compat.v1.placeholder(tf.float32,
                                                        shape=[None, None, len(self.char_list) + 1])
        self.loss_per_element = tf.compat.v1.nn.ctc_loss(labels=self.gt_texts, inputs=self.saved_ctc_input,
                                                         sequence_length=self.seq_len, ctc_merge_repeated=True)

        # best path decoding or beam search decoding
        if self.decoder_type == DecoderType.BestPath:
            self.decoder = tf.nn.ctc_greedy_decoder(inputs=self.ctc_in_3d_tbc, sequence_length=self.seq_len)
        elif self.decoder_type == DecoderType.BeamSearch:
            self.decoder = tf.nn.ctc_beam_search_decoder(inputs=self.ctc_in_3d_tbc, sequence_length=self.seq_len,
                                                         beam_width=50)
        # word beam search decoding 
        elif self.decoder_type == DecoderType.WordBeamSearch:
            # prepare information about language (dictionary, characters in dataset, characters forming words)
            chars = ''.join(self.char_list)
            word_chars = ''.join(self.char_list[:-1])
            corpus = open('corpus.txt', encoding="utf8").read()
            # decode using the "Words" mode of word beam search
            from word_beam_search import WordBeamSearch
            self.decoder = WordBeamSearch(25, 'Words', 0.0, corpus.encode('utf8'), chars.encode('utf8'),
                                          word_chars.encode('utf8'))

            # the input to the decoder must have softmax already applied
            self.wbs_input = tf.nn.softmax(self.ctc_in_3d_tbc, axis=2)

    def setup_tf(self) -> Tuple[tf.compat.v1.Session, tf.compat.v1.train.Saver]:
        """Initialize TF."""
        print('Python: ' + sys.version)
        print('Tensorflow: ' + tf.__version__)
        sess = tf.compat.v1.Session()  # TF session

        saver = tf.compat.v1.train.Saver(max_to_keep=1)  # saver saves model to file
        model_dir = '../model/'
        latest_snapshot = tf.train.latest_checkpoint(model_dir)  # is there a saved model?

        # if model must be restored (for inference), there must be a snapshot
        if self.must_restore and not latest_snapshot:
            raise Exception('No saved model found in: ' + model_dir)

        # load saved model if available
        if latest_snapshot:
            print('Init with stored values from ' + latest_snapshot)
            saver.restore(sess, latest_snapshot)
        else:
            print('Init with new values')
            sess.run(tf.compat.v1.global_variables_initializer())

        return sess, saver

    def to_sparse(self, texts: List[str]) -> Tuple[List[List[int]], List[int], List[int]]:
        """Put ground truth texts into sparse tensor for ctc_loss."""
        indices = []
        values = []
        shape = [len(texts), 0]  # last entry must be max(labelList[i])

        # go over all texts
        for batchElement, text in enumerate(texts):
            # convert to string of label (i.e. class-ids)
            label_str = [self.char_list.index(c) for c in text]
            # sparse tensor must have size of max. label-string
            if len(label_str) > shape[1]:
                shape[1] = len(label_str)
            # put each label into sparse tensor
            for i, label in enumerate(label_str):
                indices.append([batchElement, i])
                values.append(label)

        return indices, values, shape

    def decoder_output_to_text(self, ctc_output: tuple, batch_size: int) -> List[str]:
        """Extract texts from output of CTC decoder."""

        # word beam search: already contains label strings
        if self.decoder_type == DecoderType.WordBeamSearch:
            label_strs = ctc_output

        # TF decoders: label strings are contained in sparse tensor
        else:
            # ctc returns tuple, first element is SparseTensor
            decoded = ctc_output[0][0]

            # contains string of labels for each batch element
            label_strs = [[] for _ in range(batch_size)]

            # go over all indices and save mapping: batch -> values
            for (idx, idx2d) in enumerate(decoded.indices):
                label = decoded.values[idx]
                batch_element = idx2d[0]  # index according to [b,t]
                label_strs[batch_element].append(label)

        # map labels to chars for all batch elements
        # return [''.join([self.char_list[c] for c in labelStr]) for labelStr in label_strs]
        output_dec = [''.join([self.char_list[c] for c in labelStr]) for labelStr in label_strs]
        final_output = []

        for item in output_dec:
          final_output.append(item)

        return final_output

    def train_batch(self, batch: Batch) -> float:
        """Feed a batch into the NN to train it."""
        num_batch_elements = len(batch.imgs)
        max_text_len = batch.imgs[0].shape[0] // 4
        sparse = self.to_sparse(batch.gt_texts)
        eval_list = [self.optimizer, self.loss]
        feed_dict = {self.input_imgs: batch.imgs, self.gt_texts: sparse,
                     self.seq_len: [max_text_len] * num_batch_elements, self.is_train: True}
        loss_val = self.sess.run(eval_list, feed_dict)
        self.batches_trained += 1
        return loss_val

    @staticmethod
    def dump_nn_output(rnn_output: np.ndarray) -> None:
        """Dump the output of the NN to CSV file(s)."""
        dump_dir = './dump/'
        if not os.path.isdir(dump_dir):
            os.mkdir(dump_dir)

        # iterate over all batch elements and create a CSV file for each one
        max_t, max_b, max_c = rnn_output.shape
        for b in range(max_b):
            csv = ''
            for t in range(max_t):
                for c in range(max_c):
                    csv += str(rnn_output[t, b, c]) + ';'
                csv += '\n'
            fn = dump_dir + 'rnnOutput_' + str(b) + '.csv'
            print('Write dump of NN to file: ' + fn)
            with open(fn, 'w') as f:
                f.write(csv)

    def infer_batch(self, batch: Batch, calc_probability: bool = False, probability_of_gt: bool = False):
        """Feed a batch into the NN to recognize the texts."""

        # decode, optionally save RNN output
        num_batch_elements = len(batch.imgs)

        # put tensors to be evaluated into list
        eval_list = []

        if self.decoder_type == DecoderType.WordBeamSearch:
            eval_list.append(self.wbs_input)
        else:
            eval_list.append(self.decoder)

        if self.dump or calc_probability:
            eval_list.append(self.ctc_in_3d_tbc)

        # sequence length depends on input image size (model downsizes width by 4)
        max_text_len = batch.imgs[0].shape[0] // 4

        # dict containing all tensor fed into the model
        feed_dict = {self.input_imgs: batch.imgs, self.seq_len: [max_text_len] * num_batch_elements,
                     self.is_train: False}

        # evaluate model
        eval_res = self.sess.run(eval_list, feed_dict)
        # TF decoders: decoding already done in TF graph
        if self.decoder_type != DecoderType.WordBeamSearch:
            decoded = eval_res[0]
        # word beam search decoder: decoding is done in C++ function compute()
        else:
            decoded = self.decoder.compute(eval_res[0])

        # map labels (numbers) to character string
        texts = self.decoder_output_to_text(decoded, num_batch_elements)

        # feed RNN output and recognized text into CTC loss to compute labeling probability
        probs = None
        if calc_probability:
            sparse = self.to_sparse(batch.gt_texts) if probability_of_gt else self.to_sparse(texts)
            ctc_input = eval_res[1]
            eval_list = self.loss_per_element
            feed_dict = {self.saved_ctc_input: ctc_input, self.gt_texts: sparse,
                         self.seq_len: [max_text_len] * num_batch_elements, self.is_train: False}
            loss_vals = self.sess.run(eval_list, feed_dict)
            probs = np.exp(-loss_vals)

        # dump the output of the NN to CSV file(s)
        if self.dump:
            self.dump_nn_output(eval_res[1])

        return texts, probs

    def save(self) -> None:
        """Save model to file."""
        self.snap_ID += 1
        self.saver.save(self.sess, '../model/snapshot', global_step=self.snap_ID)


def infer(model: Model, fn_img: Path) -> tuple[list[str], str]:
    """Recognizes text in image provided by file path."""
    img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)

    assert img is not None

    preprocessor = Preprocessor(img_size, dynamic_width=True, padding=16)
    img = preprocessor.process_img(img)

    batch = Batch([img], None, 1)
    recognized, probability = model.infer_batch(batch, True)
    recognized_corrected = YandexSpeller().spelled(recognized[0])
    return recognized, recognized_corrected


def run_model(decoder, img_file):
    char_list = list('абвгдежзийклмнопрстуфхцчшщъыьэюя ')
    decoder_mapping = {'bestpath': DecoderType.BestPath,
                     'beamsearch': DecoderType.BeamSearch,
                     'wordbeamsearch': DecoderType.WordBeamSearch}
    decoder_type = decoder_mapping[decoder]

    model = Model(char_list, decoder_type, must_restore=True)
    recognized, recognized_corrected = infer(model, img_file)
    return recognized, recognized_corrected

def bot_infer(model: Model, fn_img) -> tuple[list[str], str]:
    """Recognizes text in image provided by file path."""

    try:
        arr = np.asarray(bytearray(fn_img.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    except:
        img = fn_img


    #img = cv2.resize(img, (0, 0), fx=1.5, fy=1.5)
    # thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # img = thresh

    # cv2.imshow("123", img)
    # cv2.waitKey(0)

    assert img is not None

    preprocessor = Preprocessor(img_size, dynamic_width=True, padding=16)
    img = preprocessor.process_img(img)

    batch = Batch([img], None, 1)
    recognized, probability = model.infer_batch(batch, True)
    recognized_corrected = YandexSpeller().spelled(recognized[0])
    return recognized, recognized_corrected

def bot_run_model(model, img_file):
    recognized, recognized_corrected = bot_infer(model, img_file)
    return recognized, recognized_corrected


def run_model_train(decoder):
    char_list = list(" !\"#&'()[]«»*+,-./0123456789:;?АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя")
    decoder_mapping = {'bestpath': DecoderType.BestPath,
                     'beamsearch': DecoderType.BeamSearch,
                     'wordbeamsearch': DecoderType.WordBeamSearch}
    decoder_type = decoder_mapping[decoder]

    from pathlib import Path
    loader = DataLoader(Path('../images/'), 16)

    model = Model(char_list, decoder_type, must_restore=True)

    train(model, loader)

#
# def train(model, loader):
#     """ Train the neural network """
#     epoch = 0  # Number of training epochs since start
#     bestCharErrorRate = float('inf')  # Best valdiation character error rate
#     noImprovementSince = 0  # Number of epochs no improvement of character error rate occured
#     earlyStopping = 8  # Stop training after this number of epochs without improvement
#     batchNum = 0
#
#     totalEpoch = len(loader.train_samples)//5500
#
#     while True:
#         epoch += 1
#         print('Epoch:', epoch, '/', totalEpoch)
#
#         # Train
#         print('Train neural network')
#         loader.train_set()
#         while loader.has_next():
#             batchNum += 1
#             iterInfo = loader.get_iterator_info()
#             batch = loader.get_next()
#             loss = model.train_batch(batch, batchNum)
#             print('Batch:', iterInfo[0], '/', iterInfo[1], 'Loss:', loss)
#
#         # Validate
#         charErrorRate, textLineAccuracy, wordErrorRate = validate(model, loader)
#         cer_summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(
#             tag='charErrorRate', simple_value=charErrorRate)])  # Tensorboard: Track charErrorRate
#
#         text_line_summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(
#             tag='textLineAccuracy', simple_value=textLineAccuracy)])  # Tensorboard: Track textLineAccuracy
#
#         wer_summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(
#             tag='wordErrorRate', simple_value=wordErrorRate)])  # Tensorboard: Track wordErrorRate
#
#
#         # If best validation accuracy so far, save model parameters
#         if charErrorRate < bestCharErrorRate:
#             print('Character error rate improved, save model')
#             bestCharErrorRate = charErrorRate
#             noImprovementSince = 0
#             model.save()
#             open("../model/accuracy.txt", 'w').write(
#                 'Validation character error rate of saved model: %f%%' % (charErrorRate*100.0))
#         else:
#             print('Character error rate not improved')
#             noImprovementSince += 1
#
#         # Stop training if no more improvement in the last x epochs
#         if noImprovementSince >= earlyStopping:
#             print('No more improvement since %d epochs. Training stopped.' %
#                   earlyStopping)
#             break
#
#
# def wer(r, h):
#     """
#     Calculation of WER with Levenshtein distance.
#
#     Works only for iterables up to 254 elements (uint8).
#     O(nm) time ans space complexity.
#
#     Parameters
#     ----------
#     r : list
#     h : list
#
#     Returns
#     -------
#     int
#
#     Examples
#     --------
#     >>> wer("who is there".split(), "is there".split())
#     1
#     >>> wer("who is there".split(), "".split())
#     3
#     >>> wer("".split(), "who is there".split())
#     3
#     """
#     # initialisation
#     d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8)
#     d = d.reshape((len(r)+1, len(h)+1))
#     for i in range(len(r)+1):
#         for j in range(len(h)+1):
#             if i == 0:
#                 d[0][j] = j
#             elif j == 0:
#                 d[i][0] = i
#
#     # computation
#     for i in range(1, len(r)+1):
#         for j in range(1, len(h)+1):
#             if r[i-1] == h[j-1]:
#                 d[i][j] = d[i-1][j-1]
#             else:
#                 substitution = d[i-1][j-1] + 1
#                 insertion = d[i][j-1] + 1
#                 deletion = d[i-1][j] + 1
#                 d[i][j] = min(substitution, insertion, deletion)
#     return d[len(r)][len(h)]
#
#
# def validate(model, loader):
#     """ Validate neural network """
#     print('Validate neural network')
#     loader.validationSet()
#     numCharErr = 0
#     numCharTotal = 0
#     numWordOK = 0
#     numWordTotal = 0
#
#     totalCER = []
#     totalWER = []
#     while loader.hasNext():
#         iterInfo = loader.getIteratorInfo()
#         print('Batch:', iterInfo[0], '/', iterInfo[1])
#         batch = loader.getNext()
#         recognized = model.infer_batch(batch)
#
#         print('Ground truth -> Recognized')
#         for i in range(len(recognized)):
#             numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0
#             numWordTotal += 1
#             dist = editdistance.eval(recognized[i], batch.gtTexts[i])
#
#             currCER = dist/max(len(recognized[i]), len(batch.gtTexts[i]))
#             totalCER.append(currCER)
#
#             currWER = wer(recognized[i].split(), batch.gtTexts[i].split())
#             totalWER.append(currWER)
#
#             numCharErr += dist
#             numCharTotal += len(batch.gtTexts[i])
#             print('[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' +
#                   batch.gtTexts[i] + '"', '->', '"' + recognized[i] + '"')
#
#     # Print validation result
#     try:
#         charErrorRate = sum(totalCER)/len(totalCER)
#         wordErrorRate = sum(totalWER)/len(totalWER)
#         textLineAccuracy = numWordOK / numWordTotal
#     except ZeroDivisionError:
#         charErrorRate = 0
#         wordErrorRate = 0
#         textLineAccuracy = 0
#     print('Character error rate: %f%%. Text line accuracy: %f%%. Word error rate: %f%%' %
#           (charErrorRate*100.0, textLineAccuracy*100.0, wordErrorRate*100.0))
#     return charErrorRate, textLineAccuracy, wordErrorRate

def write_summary(char_error_rates: List[float], word_accuracies: List[float]) -> None:
    """Writes training summary file for NN."""
    with open("../model/summary.json", 'w') as f:
        json.dump({'charErrorRates': char_error_rates, 'wordAccuracies': word_accuracies}, f)

def train(model, loader):
    """Trains NN."""
    epoch = 0  # number of training epochs since start
    summary_char_error_rates = []
    summary_word_accuracies = []
    preprocessor = Preprocessor(img_size, dynamic_width=True, padding=16)
    best_char_error_rate = float('inf')  # best validation character error rate
    no_improvement_since = 0  # number of epochs no improvement of character error rate occurred
    # stop training after this number of epochs without improvement

    while True:
        epoch += 1
        print('Epoch:', epoch)

        # train
        print('Train NN')
        loader.train_set()
        while loader.has_next():
            iter_info = loader.get_iterator_info()
            batch = loader.get_batch()
            batch = preprocessor.process_batch(batch)
            loss = model.train_batch(batch)
            print(f'Epoch: {epoch} Batch: {iter_info[0]}/{iter_info[1]} Loss: {loss}')

        # validate
        char_error_rate, word_accuracy = validate(model, loader, False)

        # write summary
        summary_char_error_rates.append(char_error_rate)
        summary_word_accuracies.append(word_accuracy)
        write_summary(summary_char_error_rates, summary_word_accuracies)

        # if best validation accuracy so far, save model parameters
        if char_error_rate < best_char_error_rate:
            print('Character error rate improved, save model')
            best_char_error_rate = char_error_rate
            no_improvement_since = 0
            model.save()
        else:
            print(f'Character error rate not improved, best so far: {char_error_rate * 100.0}%')
            no_improvement_since += 1

        # stop training if no more improvement in the last x epochs
        if no_improvement_since >= 25:
            print(f'No more improvement since {25} epochs. Training stopped.')
            break


def validate(model: Model, loader, line_mode: bool) -> Tuple[float, float]:
    """Validates NN."""
    print('Validate NN')
    loader.validation_set()
    preprocessor = Preprocessor(img_size, dynamic_width=True, padding=16)
    num_char_err = 0
    num_char_total = 0
    num_word_ok = 0
    num_word_total = 0
    while loader.has_next():
        iter_info = loader.get_iterator_info()
        print(f'Batch: {iter_info[0]} / {iter_info[1]}')
        batch = loader.get_batch()
        batch = preprocessor.process_batch(batch)
        recognized, _ = model.infer_batch(batch)

        print('Ground truth -> Recognized')
        for i in range(len(recognized)):
            num_word_ok += 1 if batch.gt_texts[i] == recognized[i] else 0
            num_word_total += 1
            dist = editdistance.eval(recognized[i], batch.gt_texts[i])
            num_char_err += dist
            num_char_total += len(batch.gt_texts[i])
            print('[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' + batch.gt_texts[i] + '"', '->',
                  '"' + recognized[i] + '"')

    # print validation result
    char_error_rate = num_char_err / num_char_total
    word_accuracy = num_word_ok / num_word_total
    print(f'Character error rate: {char_error_rate * 100.0}%. Word accuracy: {word_accuracy * 100.0}%.')
    return char_error_rate, word_accuracy



@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
