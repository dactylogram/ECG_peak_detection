import numpy as np
import torch.utils.data as data

feature_shape = 2048

class Test_Loader(data.Dataset):
    def __init__(self, set_dict):
        self.list_feature = set_dict['feature']
        self.list_target = set_dict['target']
        self.iteration = len(self.list_feature)
        # print('Test ... No. of Records :', self.iteration)

    def __len__(self):
        return self.iteration

    def __getitem__(self, index):
        feature = self.list_feature[index]
        target = self.list_target[index]

        padding = -feature.shape[0]%feature_shape # cropping
        if padding != 0:
            feature = np.pad(feature, ((0, padding), (0, 0)))
            target = np.pad(target, (0, padding))

        X = np.swapaxes(feature, 0, 1)
        y = target[np.newaxis, :]

        return X, y

def Test_Generator(set_dict):
    data_loader = Test_Loader(set_dict)
    data_generator = data.DataLoader(data_loader, batch_size=1, shuffle=False)
    return data_generator
