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

import numpy as np
from sklearn.utils import check_random_state


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
