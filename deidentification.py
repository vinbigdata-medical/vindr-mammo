from pydicom.dataset import FileDataset
import hashlib
import pandas as pd

# load keeping tags
TAGS_DF = pd.read_csv("./tags.csv")


def deidentify(dicom: FileDataset) -> FileDataset:
    """
    keep tags from a predefined list and remove all remaining tags
    hash StudyInstanceUID, SeriesInstanceUID, SOPInstanceUID

    args:
        dicom: loaded dicom image via pydicom.dcmread(<<path>>)

    return masked dicom image
    """
    dicom.file_meta.ImplementationClassUID = "1.2.3.4"
    dicom.file_meta.MediaStorageSOPClassUID = "1.2.3.4"
    img_id = dicom.file_meta.MediaStorageSOPInstanceUID
    masked_img_id = hash_fn(img_id)
    dicom.file_meta.MediaStorageSOPInstanceUID = masked_img_id
    
    series_id = dicom.SeriesInstanceUID
    masked_series_id = hash_fn(series_id)
    dicom.SeriesInstanceUID = masked_series_id
    
    study_id = dicom.StudyInstanceUID
    masked_study_id = hash_fn(study_id)
    dicom.StudyInstanceUID = masked_study_id
    
    img_id = dicom.SOPInstanceUID
    masked_img_id = hash_fn(img_id)
    dicom.SOPInstanceUID = masked_img_id
    
    dicom_keys = list(dicom.keys())
    for k in dicom_keys:
        if k not in TAGS_DF.Decimal.values:
            dicom.pop(k, None)
    return dicom


def hash_fn(text: str) -> str:
    hash_obj = hashlib.md5(text.encode())
    return str(hash_obj.hexdigest())



