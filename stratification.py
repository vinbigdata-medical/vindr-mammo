"""Implementation of the Iterative Stratification algorithm (Sechidis et al. 2011)
    with a slight modification that allows a data sample to have multiple instance of a labels
    
    The code is adapted from :
    https://github.com/trent-b/iterative-stratification/blob/master/iterstrat/ml_stratifiers.py
    
    Sechidis K., Tsoumakas G., Vlahavas I. (2011) On the Stratification of
    Multi-Label Data. In: Gunopulos D., Hofmann T., Malerba D., Vazirgiannis M.
    (eds) Machine Learning and Knowledge Discovery in Databases. ECML PKDD
    2011. Lecture Notes in Computer Science, vol 6913. Springer, Berlin,
    Heidelberg.
    """

import json
import pandas as pd
import numpy as np
from sklearn.utils import check_random_state
from collections import Counter 
import copy


LABELS = [
    "Ankylosis", # (Dính khớp) 
    "Cortical destruction", # (phá hủy vỏ xương) 
    "Disc space narrowing", # (hẹp khe đĩa đệm) 
    "Enthesophytes", # (cầu xương) 
    "Foraminal stenosis", # (hẹp lỗ tiếp hợp)
    "Fracture", # (gãy xương)
    "Ground glass density", # (Kính mờ)
    "Lytic lesion", # (tổn thương dạng tiêu xương)
    "Mixed lesion", # (tổn thương đặc/tiêu xương hỗn hợp)
    "Osteophytes", # (gai xương)
    "Osteoporosis", # (Loãng xương)
    "Sclerotic lesion", # (tổn thương dạng đặc xương)
    "Sclerotic rim", # (Viền đặc xương) 
    "Spondylolysthesis", # (Trươt đốt sống)
    "Subchondral sclerosis", # (Đặc xương dưới sụn)
    "Surgical implant", # (Vật liệu phẫu thuật)
    "Vertebral collapse", # (Xẹp đốt sống)
    "Foreign body", # (Dị vật)
    "Other lesions", # (các tổn thương khác)
]

NAME2INDEX = dict(zip(LABELS, range(len(LABELS))))


class IterativeStratification:

    def __init__(self, seed=0, max_studies=None):
        self.random_state = check_random_state(seed)
        self.max_studies = max_studies
        
    def stratify(self, labels: np.ndarray, r: np.ndarray):
        """
        args:
            -- labels: np array of shape (N,M), where each row represent a data sample with M labels
            -- r: indicates portion of data in each subset. Note that sum of all elements mus be one
        """
        assert r.sum() == 1., f"sum of all ratio mus be 1, got {r}"
        
        random_state = self.random_state
        
        n_samples = labels.shape[0]
        if self.max_studies:
            n_samples = min(self.max_studies, n_samples)
        test_folds = np.zeros(n_samples, dtype=int)

        # shuffle 
        shuffled_ids = np.arange(n_samples)
        random_state.shuffle(shuffled_ids)
        labels = labels[shuffled_ids]
        # Calculate the desired number of examples at each subset
        c_folds = np.ceil(r * n_samples)

        # Calculate the desired number of examples of each label at each subset
        c_folds_labels = np.outer(r, labels.sum(axis=0))

        labels_not_processed_mask = np.ones(n_samples, dtype=bool)

        while np.any(labels_not_processed_mask):
            # Find the label with the fewest (but at least one) remaining examples,
            # breaking ties randomly
            num_labels = labels[labels_not_processed_mask].sum(axis=0)

            # Handle case where only all-zero labels are left by distributing
            # across all folds as evenly as possible (not in original algorithm but
            # mentioned in the text). (By handling this case separately, some
            # code redundancy is introduced; however, this approach allows for
            # decreased execution time when there are a relatively large number
            # of all-zero labels.)
            if num_labels.sum() == 0:
                sample_idxs = np.where(labels_not_processed_mask)[0]

                for sample_idx in sample_idxs:
                    fold_idx = np.where(c_folds == c_folds.max())[0]

                    if fold_idx.shape[0] > 1:
                        fold_idx = fold_idx[random_state.choice(fold_idx.shape[0])]

                    test_folds[shuffled_ids[sample_idx]] = fold_idx
                    c_folds[fold_idx] -= 1

                break

            label_idx = np.where(num_labels == num_labels[np.nonzero(num_labels)].min())[0]
            if label_idx.shape[0] > 1:
                label_idx = label_idx[random_state.choice(label_idx.shape[0])]

            sample_idxs = np.where(np.logical_and(
                labels[:, label_idx].flatten(), 
                labels_not_processed_mask
            ))[0]
            
            # TODO sort sample idxs by label weight
            for sample_idx in sample_idxs:
                # Find the subset(s) with the largest number of desired examples
                # for this label, breaking ties by considering the largest number
                # of desired examples, breaking further ties randomly
                label_folds = c_folds_labels[:, label_idx]
                fold_idx = np.where(label_folds == label_folds.max())[0]

                if fold_idx.shape[0] > 1:
                    temp_fold_idx = np.where(c_folds[fold_idx] ==
                                            c_folds[fold_idx].max())[0]
                    fold_idx = fold_idx[temp_fold_idx]

                    if temp_fold_idx.shape[0] > 1:
                        fold_idx = fold_idx[random_state.choice(temp_fold_idx.shape[0])]

                test_folds[shuffled_ids[sample_idx]] = fold_idx
                labels_not_processed_mask[sample_idx] = False

                # Update desired number of examples
                c_folds_labels[fold_idx] -= labels[sample_idx]
                c_folds[fold_idx] -= 1
        return test_folds

    def convert_label(cls, label):
        """
        convert labels from json format into numpy array with one study per row, along with rad index, rad id
        
        return X, y, rad_index, rad_id, n_imgs
            X: list of study ids of length N
            y: np.array of shape (N,L+2)
            rad_ids: list of all rad ids
            rad_indices: np.array of length N indicate index (in rad_ids) of rad for each study
            n_imgs: list of length N - number of image in each studies
        
        NOTE: count No finding images, all images in a study
        """
        
        
        X, y, rad_per_study, n_imgs = [], [], [], []
        n_labels = len(LABELS)
        # flatten labels by image
        for study_id, study_info in label.items():
            rad_per_study.append(study_info["rad_id"])
            X.append(study_id)
            
            # +2 for nf counter and all img counter
            label_count = np.zeros((n_labels + 2,),dtype=np.int)
            
            series_labels = study_info["labels"]
            for series_id, series_label in series_labels.items():
                for image_labels in series_label["labels"].values():
                    label_count[-1] += 1
                    if "No finding" in image_labels["image_impression"]:
                        label_count[-2] += 1
                        # assert len(image_labels["image_impression"]) == 0, \
                        #     f"study {study_id}: invalid impression, got {image_labels['image_impression']}"
                        # assert len(image_labels["class_list"]) == 0, "nf image has finding"
                    else:
                        for classes in image_labels["class_list"]:
                            for _class in classes:
                                label_count[NAME2INDEX[_class]] += 1
                    
            y.append(label_count)
        y = np.stack(y, axis=0)
        
        
        rad_ids = list(set(rad_per_study))
        rad_mapper = dict(zip(rad_ids, range(len(rad_ids))))
        rad_indices = np.array([rad_mapper[rad] for rad in rad_per_study])
        
        return X, y, rad_ids, rad_indices

    def stratify_json(cls, json_path, r=np.array([0.75,0.25])):
        """
        NOTE: ignore Osteoporosis
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        X, y, rad_ids, rad_indices = cls.convert_label(data) 
        print(len(X), len(y), len(rad_indices))
        print(rad_ids)
        print(y.sum(axis=0))
        print(Counter(rad_indices))
        
        # remove Osteoporosis
        _LABELS = copy.deepcopy(LABELS)
        osteoporosis_id = NAME2INDEX["Osteoporosis"]
        _LABELS.pop(osteoporosis_id)
        label_ids = list(np.arange(y.shape[1]))
        label_ids.pop(osteoporosis_id)
        label_ids = np.array(label_ids)
        y = y[:,label_ids]
        print("#########",y.shape)
        # convert rad_indices to onehot for stratification
        n_rads = len(rad_ids)
        rad_oh = np.eye(n_rads)[rad_indices]
    
        split_results = cls.stratify(np.concatenate([y,rad_oh], axis=1), r=r)
        # split_results = split_results[:5000]
        for i, name in enumerate(["40", "60"]):
            sample_idxs = np.where(split_results==i)[0]
            label_stats = y[sample_idxs].sum(axis=0)
            rad_per_fold = rad_oh[sample_idxs].sum(axis=0)
            print(f"{name} set: num sample {len(sample_idxs)} \n labels {label_stats} \n rad {rad_per_fold}")
        
            # save results
            study_id_per_set = [X[idx] for idx in sample_idxs]
            label_per_set = {k: data[k] for k in study_id_per_set}
            target_path = json_file.replace(".json", f"_{name}.json")
            with open(target_path, "w") as f:
                json.dump(label_per_set, f, indent=4)
            
            # export stats
            box_stats_df = pd.DataFrame({"lesion": _LABELS + ["nf", "img"], "n_boxes": list(label_stats)})
            box_stats_df.to_csv(json_file.replace(".json", f"_{name}_stats.csv"), index=False)
        
if __name__ == "__main__":
    import pandas as pd
    # for seed in range(1950,2000):
        # print("\n\n", seed)
    seed = 42
    # json_file = "/mnt/DATA/bonie/labels/3999_labeled_cases_31_12_2020_finding.json"
    # json_file = "/mnt/DATA/bonie/labels/corrected_20210204_finding_size.json"
    json_file = "/mnt/DATA/bonie/labels/corrected_20210218_2333_labels_5299_train.json"



    spliter = IterativeStratification(seed, 5000)
    spliter.stratify_json(json_file, np.array([0.4,0.6]))