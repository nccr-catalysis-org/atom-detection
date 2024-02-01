from enum import Enum


class Catalyst(Enum):
    Pt = 'Pt'
    Fe = 'Fe'

    def __str__(self):
        return str(self.value)


class Method(Enum):
    DL = 'DL'
    CV = 'CV'
    TEM = 'TEMImageNet'

    def __str__(self):
        return str(self.value)


class Split:
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


class Columns:
    FILENAME = 'Filename'
    LABEL = 'Label'
    SPLIT = 'Split'


class CropsColumns:
    FILENAME = 'Filename'
    ORIGINAL = 'Original'
    X = 'X'
    Y = 'Y'
    LABEL = 'Label'
    SPLIT = 'Split'


class BoxColumns:
    FILENAME = 'Filename'
    X1 = 'X1'
    X2 = 'X2'
    Y1 = 'Y1'
    Y2 = 'Y2'
    LABEL = 'Label'
    SPLIT = 'Split'


class ProbsColumns:
    FILENAME = 'Filename'
    ORIGINAL = 'Original'
    LABEL = 'Label'
    SPLIT = 'Split'


class ModelArgs(str, Enum):
    BASICCNN = 'basic'
    RESNET18 = 'resnet18'
