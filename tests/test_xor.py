import unittest

import tensorflow as tf

from xor_parity import generate_train_data, dataset_from_gen

tf.enable_eager_execution()


class TestXor(unittest.TestCase):

    @staticmethod
    def _collect_bits(generator):
        res = []
        while True:
            try:
                e = next(generator)
                res.append(e[0].flatten())
            except StopIteration:
                break
        return res

    def test_generate_bits_all_warmup(self):
        '''
        test that all generated bits should have length max_length, and [0:-2] should all be 0
        '''
        res = self._collect_bits(generate_train_data(count=5, max_length=5, warmup=5))
        self.assertEqual(len(res), 5)
        for bits in res:
            self.assertEqual(len(bits), 5)
            self.assertTrue(all([e == 0 for e in bits[0:-2]]))

    def test_generate_bits_sorted_by_length(self):
        '''
        test that the bits are sorted by effective length
        '''
        res = self._collect_bits(generate_train_data(count=10, max_length=10, warmup=0))
        self.assertEqual(len(res), 10)
        for bits in res:
            self.assertEqual(len(bits), 10)
            print(bits)

        # todo: how to test they are ordered by effective length

    def test_dataset(self):
        generator = generate_train_data(count=10, max_length=5, warmup=5)
        dataset = dataset_from_gen(generator, batch_size=2)
        for i in dataset:
            data, label = i
            self.assertEqual(data.shape, (2, 5, 1))
            self.assertEqual(label.shape, (2, 1))
