import numpy
import random
import matplotlib.pyplot as plt


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
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # 5
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # 6
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

        source_bipolar = [
            [  # E
                # 1  2   3   4   5   6   7   8   9  10
                [1,  1,  1,  1,  1,  1,  1,  1,  1,  1],    # 1
                [1,  1,  1,  1,  1,  1,  1,  1,  1,  1],    # 2
                [1,  1, -1, -1, -1, -1, -1, -1, -1, -1],    # 3
                [1,  1, -1, -1, -1, -1, -1, -1, -1, -1],    # 4
                [1,  1,  1,  1,  1,  1,  1,  1,  1, -1],    # 5
                [1,  1,  1,  1,  1,  1,  1,  1,  1, -1],    # 6
                [1,  1, -1, -1, -1, -1, -1, -1, -1, -1],    # 7
                [1,  1, -1, -1, -1, -1, -1, -1, -1, -1],    # 8
                [1,  1,  1,  1,  1,  1,  1,  1,  1,  1],    # 9
                [1,  1,  1,  1,  1,  1,  1,  1,  1,  1],    # 10
            ],
            [  # O
                # 1  2   3   4   5   6   7   8   9  10
                [1,  1,  1,  1,  1,  1,  1,  1,  1,  1],    # 1
                [1,  1,  1,  1,  1,  1,  1,  1,  1,  1],    # 2
                [1,  1, -1, -1, -1, -1, -1, -1,  1,  1],    # 3
                [1,  1, -1, -1, -1, -1, -1, -1,  1,  1],    # 4
                [1,  1, -1, -1, -1, -1, -1, -1,  1,  1],    # 5
                [1,  1, -1, -1, -1, -1, -1, -1,  1,  1],    # 6
                [1,  1, -1, -1, -1, -1, -1, -1,  1,  1],    # 7
                [1,  1, -1, -1, -1, -1, -1, -1,  1,  1],    # 8
                [1,  1,  1,  1,  1,  1,  1,  1,  1,  1],    # 9
                [1,  1,  1,  1,  1,  1,  1,  1,  1,  1],    # 10
            ],
            [  # [sha]
                # 1  2   3   4   5   6   7   8   9  10
                [1,  1, -1, -1, -1, -1, -1, -1,  1,  1],    # 1
                [1,  1, -1, -1, -1, -1, -1, -1,  1,  1],    # 2
                [1,  1, -1, -1,  1,  1, -1, -1,  1,  1],    # 3
                [1,  1, -1, -1,  1,  1, -1, -1,  1,  1],    # 4
                [1,  1, -1, -1,  1,  1, -1, -1,  1,  1],    # 5
                [1,  1, -1, -1,  1,  1, -1, -1,  1,  1],    # 6
                [1,  1, -1, -1,  1,  1, -1, -1,  1,  1],    # 7
                [1,  1,  1,  1,  1,  1,  1,  1,  1,  1],    # 8
                [1,  1,  1,  1,  1,  1,  1,  1,  1,  1],    # 9
                [1,  1,  1,  1,  1,  1,  1,  1,  1,  1],    # 10
            ]
        ]
        result = []
        for item in source_bipolar:
            result.append(numpy.array(item, dtype=int))
        return result


class NoiseGenerator:
    @staticmethod
    def dot_noise(img, prob: float, samples: int = 1):
        if samples < 0:
            raise ValueError
        if samples == 1:
            return numpy.where(numpy.random.rand(*img.shape) < prob, -img, img)
            # return numpy.where(numpy.random.rand(*img.shape) < prob, 1 - img, img)
        result = []
        for i in range(samples):
            result.append(numpy.where(numpy.random.rand(*img.shape) < prob, -img, img))
            # result.append(numpy.where(numpy.random.rand(*img.shape) < prob, 1 - img, img))
        return result


class HopfieldNetwork:
    weights = None

    def __init__(self, samples: list):
        self.weights = numpy.zeros(shape=(100, 100), dtype=int)
        for item in samples:
            # temp = numpy.reshape(item, newshape=(item.shape[0], 1))
            self.weights += numpy.matmul(item, numpy.transpose(item))
        self.weights[numpy.eye(self.weights.shape[0], dtype=bool)] = 0

    def sync_step(self, in_mx):
        return numpy.where(numpy.matmul(self.weights, in_mx) > 0, 1, -1)

    def async_step(self, in_mx):
        t_weights = numpy.eye(self.weights.shape[0])
        index = random.randrange(self.weights.shape[0])
        t_weights[index] = self.weights[index]
        return numpy.where(numpy.matmul(t_weights, in_mx) > 0, 1, -1)


def lab():
    params = [0.2, 4]
    _labels = [f"Dot noise probability [0.2]: ", f"Number of test samples [4]: "]
    _p_types = [float, int]

    for i in range(len(params)):
        try:
            temp = _p_types[i](input(_labels[i]))
            params[i] = temp if temp > 0 else params[i]
        except ValueError:
            continue

    dataset = SampleGenerator.get_samples()
    flattened = []
    for item in dataset:
        flattened.append(numpy.reshape(item, newshape=(item.size, 1)))
    network = HopfieldNetwork(flattened)

    test_samples = []
    for item in flattened:
        if params[1] > 1:
            test_samples.extend(NoiseGenerator.dot_noise(item, params[0], params[1]))
        else:
            test_samples.append(NoiseGenerator.dot_noise(item, params[0]))

    plot_content = []
    for item in test_samples:
        prev_state = item
        for i in range(int(10e4)):
            current = network.sync_step(prev_state)
            if numpy.all(prev_state == current):
                break
            prev_state = current
        else:
            print(f"Step limit exceeded")
        plot_content.append(
            tuple((
                numpy.reshape(item, newshape=(10, 10)),
                numpy.reshape(prev_state, newshape=(10, 10)))))
        # break

    figures = []
    for item in plot_content:
        fig = plt.figure()
        if type(item) is tuple:
            axes = fig.subplots(len(item), 1)
            for i in range(len(item)):
                axes[i].imshow(item[i])
                axes[i].set_axis_off()
            pass
        # axes = fig.add_subplot()
        # axes.imshow(item)
        # axes.set_axis_off()
        figures.append(fig)
    plt.show()
    #   flatten dataset and initialize Hopfield network
    pass


if __name__ == '__main__':
    lab()
