import numpy
import random


class SampleGenerator:
    @staticmethod
    def get_samples():
        source = [
            [   # E
                # 1  2  3  4  5  6  7  8  9 10
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 1
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 2
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 3
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 4
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 5
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 6
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 7
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 8
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 9
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 10
            ],
            [   # O
                # 1  2  3  4  5  6  7  8  9 10
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 1
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 2
                [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],  # 3
                [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],  # 4
                [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],  # 5
                [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],  # 6
                [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],  # 7
                [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],  # 8
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 9
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 10
            ],
            [   # [sha]
                # 1  2  3  4  5  6  7  8  9 10
                [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],  # 1
                [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],  # 2
                [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],  # 3
                [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],  # 4
                [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],  # 5
                [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],  # 6
                [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],  # 7
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 8
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 9
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 10
            ]
        ]
        result = []
        for item in source:
            result.append(numpy.array(item, dtype=int))
        return result


class HopfieldNetwork:
    weights = None

    def __init__(self, samples: list):
        self.weights = numpy.zeros(shape=(100, 100), dtype=int)
        for item in samples:
            self.weights += numpy.matmul(item, numpy.transpose(item))
        self.weights[numpy.eye(self.weights.shape[0], dtype=bool)] = 0

    def sync_step(self, in_mx):
        return numpy.where(numpy.matmul(in_mx, self.weights) > 0, 1, -1)

    def async_step(self, in_mx):
        t_weights = numpy.eye(self.weights.shape[0])
        index = random.randrange(self.weights.shape[0])
        t_weights[index] = self.weights[index]
        return numpy.where(numpy.matmul(in_mx, t_weights) > 0, 1, -1)


def lab():
    dataset = SampleGenerator.get_samples()
    #   flatten dataset and initialize Hopfield network
    pass


if __name__ == '__main__':
    lab()
