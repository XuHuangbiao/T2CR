import torch
import numpy as np
import os
import pickle
import random
import glob
# from os.path import join
from PIL import Image
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class MTLPair_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, subset, transform):
        random.seed(args.seed)
        self.onehot_labels = None
        self.label_encoder = None
        self.subset = subset
        self.transforms = transform
        # using Difficult Degree
        self.usingDD = args.usingDD
        # some flags
        self.dive_number_choosing = args.dive_number_choosing
        # file path
        self.label_path = args.label_path
        self.split_path = args.train_split
        self.split = self.read_pickle(self.split_path)
        self.label_dict = self.read_pickle(self.label_path)
        self.data_root = args.data_root
        # setting
        self.temporal_shift = [args.temporal_shift_min, args.temporal_shift_max]
        self.voter_number = args.voter_number
        self.length = args.frame_length
        # build difficulty dict ( difficulty of each action, the cue to choose exemplar)
        self.difficulties_dict = {}
        self.dive_number_dict = {}
        self.dive_class_dict = {}
        self.preprocess_class()
        if self.subset == 'test':
            self.split_path_test = args.test_split
            self.split_test = self.read_pickle(self.split_path_test)
            self.difficulties_dict_test = {}
            self.dive_number_dict_test = {}
        if self.usingDD:
            self.preprocess()
            self.check()

        self.choose_list = self.split.copy()
        if self.subset == 'test':
            self.dataset = self.split_test
        else:
            self.dataset = self.split

    def load_video(self, video_file_name, phase):
        # print(video_file_name)
        # print(os.path.join(self.data_root, str('{:02d}'.format(video_file_name[0]))))
        image_list = sorted(
            (glob.glob(os.path.join(self.data_root, str('{:02d}'.format(video_file_name[0])), '*.jpg'))))
        end_frame = self.label_dict.get(video_file_name).get('end_frame')
        if phase == 'train':
            temporal_aug_shift = random.randint(self.temporal_shift[0], self.temporal_shift[1])
            end_frame = end_frame + temporal_aug_shift
        start_frame = end_frame - self.length

        video = [Image.open(image_list[start_frame + i]) for i in range(self.length)]
        return self.transforms(video)

    def read_pickle(self, pickle_path):
        with open(pickle_path, 'rb') as f:
            pickle_data = pickle.load(f)
        return pickle_data

    def preprocess_class(self):
        dive_label = []
        for item in self.label_dict:
            dive_label.append(self.label_dict.get(item)['dive_number'])
        dive_label = list(set(dive_label))
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(dive_label)
        onehot_encoder = OneHotEncoder(sparse_output=False, categories='auto')
        self.onehot_labels = onehot_encoder.fit_transform(encoded_labels.reshape(-1, 1))
        # self.dive_class_dict = dict(zip(dive_label, encoded_labels))

    def preprocess(self):
        if self.dive_number_choosing:
            # Dive Number
            for item in self.split:
                dive_number = self.label_dict.get(item)['dive_number']
                if self.dive_number_dict.get(dive_number) is None:
                    self.dive_number_dict[dive_number] = []
                self.dive_number_dict[dive_number].append(item)

            if self.subset == 'test':
                for item in self.split_test:
                    dive_number = self.label_dict.get(item)['dive_number']
                    if self.dive_number_dict_test.get(dive_number) is None:
                        self.dive_number_dict_test[dive_number] = []
                    self.dive_number_dict_test[dive_number].append(item)
        else:
            # DD
            for item in self.split:
                difficulty = self.label_dict.get(item)['difficulty']
                if self.difficulties_dict.get(difficulty) is None:
                    self.difficulties_dict[difficulty] = []
                self.difficulties_dict[difficulty].append(item)

            if self.subset == 'test':
                for item in self.split_test:
                    difficulty = self.label_dict.get(item)['difficulty']
                    if self.difficulties_dict_test.get(difficulty) is None:
                        self.difficulties_dict_test[difficulty] = []
                    self.difficulties_dict_test[difficulty].append(item)

    def check(self):
        if self.dive_number_choosing:
            # dive_number_dict
            for key in sorted(list(self.dive_number_dict.keys())):
                file_list = self.dive_number_dict[key]
                for item in file_list:
                    assert self.label_dict[item]['dive_number'] == key

            if self.subset == 'test':
                for key in sorted(list(self.dive_number_dict_test.keys())):
                    file_list = self.dive_number_dict_test[key]
                    for item in file_list:
                        assert self.label_dict[item]['dive_number'] == key
        else:
            # difficulties_dict
            for key in sorted(list(self.difficulties_dict.keys())):
                file_list = self.difficulties_dict[key]
                for item in file_list:
                    assert self.label_dict[item]['difficulty'] == key

            if self.subset == 'test':
                for key in sorted(list(self.difficulties_dict_test.keys())):
                    file_list = self.difficulties_dict_test[key]
                    for item in file_list:
                        assert self.label_dict[item]['difficulty'] == key

        print('check done')

    def __getitem__(self, index):
        sample_1 = self.dataset[index]
        data = {}
        if self.subset == 'test':
            # test phase
            data['video'] = self.load_video(sample_1, 'test')
            data['final_score'] = self.label_dict.get(sample_1).get('final_score')
            data['difficulty'] = self.label_dict.get(sample_1).get('difficulty')
            data['completeness'] = (data['final_score'] / data['difficulty'])
            data['judge_scores'] = np.sort(self.label_dict.get(sample_1).get('judge_scores'))[2:5]
            data['var'] = np.var(data['judge_scores'])
            sample_label_encoded = self.label_encoder.transform([self.label_dict.get(sample_1).get('dive_number')])
            data['dive_class'] = self.onehot_labels[sample_label_encoded]

            if self.usingDD:
                # NOTE: using Dive Number to choose
                if self.dive_number_choosing:
                    train_file_list = self.dive_number_dict[self.label_dict[sample_1]['dive_number']]
                    random.shuffle(train_file_list)
                    choosen_sample_list = train_file_list[:self.voter_number]
                else:
                    # choose a list of sample in training_set
                    train_file_list = self.difficulties_dict[self.label_dict[sample_1]['difficulty']]
                    random.shuffle(train_file_list)
                    choosen_sample_list = train_file_list[:self.voter_number]
            else:
                train_file_list = self.choose_list
                random.shuffle(train_file_list)
                choosen_sample_list = train_file_list[:self.voter_number]

            target_list = []
            for item in choosen_sample_list:
                tmp = {}
                tmp['video'] = self.load_video(item, 'test')
                tmp['final_score'] = self.label_dict.get(item).get('final_score')
                tmp['difficulty'] = self.label_dict.get(item).get('difficulty')
                tmp['completeness'] = (tmp['final_score'] / tmp['difficulty'])
                tmp['judge_scores'] = np.sort(self.label_dict.get(item).get('judge_scores'))[2:5]
                tmp['var'] = np.var(tmp['judge_scores'])
                sample_label_encoded = self.label_encoder.transform([self.label_dict.get(item).get('dive_number')])
                tmp['dive_class'] = self.onehot_labels[sample_label_encoded]
                # print(tmp)
                target_list.append(tmp)

            return data, target_list
        else:
            # train phase
            data['video'] = self.load_video(sample_1, 'train')
            data['final_score'] = self.label_dict.get(sample_1).get('final_score')
            data['difficulty'] = self.label_dict.get(sample_1).get('difficulty')
            data['completeness'] = (data['final_score'] / data['difficulty'])
            data['judge_scores'] = np.sort(self.label_dict.get(sample_1).get('judge_scores'))[2:5]
            data['var'] = np.var(data['judge_scores'])
            sample_label_encoded = self.label_encoder.transform([self.label_dict.get(sample_1).get('dive_number')])
            data['dive_class'] = self.onehot_labels[sample_label_encoded]

            # choose a sample
            if self.usingDD:
                # did not using a pytorch sampler, using diff_dict to pick a video sample
                if self.dive_number_choosing:
                    # NOTE: using Dive Number to choose
                    file_list = self.dive_number_dict[self.label_dict[sample_1]['dive_number']].copy()
                else:
                    # all sample owning same difficulties
                    file_list = self.difficulties_dict[self.label_dict[sample_1]['difficulty']].copy()
            else:
                # randomly
                file_list = self.split.copy()
            # exclude self
            if len(file_list) > 1:
                file_list.pop(file_list.index(sample_1))
            # choosing one out
            idx = random.randint(0, len(file_list) - 1)
            sample_2 = file_list[idx]
            target = {}
            # sample 2
            target['video'] = self.load_video(sample_2, 'train')
            target['final_score'] = self.label_dict.get(sample_2).get('final_score')
            target['difficulty'] = self.label_dict.get(sample_2).get('difficulty')
            target['completeness'] = (target['final_score'] / target['difficulty'])
            target['judge_scores'] = np.sort(self.label_dict.get(sample_2).get('judge_scores'))[2:5]
            target['var'] = np.var(target['judge_scores'])
            sample_label_encoded = self.label_encoder.transform([self.label_dict.get(sample_2).get('dive_number')])
            target['dive_class'] = self.onehot_labels[sample_label_encoded]
            return data, target

    def __len__(self):
        return len(self.dataset)
