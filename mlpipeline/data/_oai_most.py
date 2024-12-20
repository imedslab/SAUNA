import argparse
import torch
from torch import Tensor
import numpy as np
import os
import cv2
import glob
import pandas as pd
from tqdm import tqdm
from sas7bdat import SAS7BDAT

# Confusion matrix
import re
import itertools
import matplotlib
from textwrap import wrap
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

oai_meta_csv_filename = "oai_meta.csv"
most_meta_csv_filename = "most_meta.csv"

oai_participants_csv_filename = "oai_participants.csv"
most_participants_csv_filename = "most_participants.csv"

oai_most_meta_csv_filename = "oai_most_meta.csv"

oai_most_all_csv_filename = "oai_most_all.csv"
oai_most_img_csv_filename = "oai_most_img_patches.csv"

STD_SZ = (128, 128)

follow_up_dict_most = {0: '00', 1: '15', 2: '30', 3: '60', 5: '84'}
visit_to_month = {"most": ["00", "15", "30", "60", "84"], "oai": [
    "00", "12", "24", "36", "48", "72", "96"]}


def read_sas7bdata_pd(fname):
    data = []
    with SAS7BDAT(fname) as f:
        for row in f:
            data.append(row)

    return pd.DataFrame(data[1:], columns=data[0])


def build_img_klg_meta_oai(img_dir, oai_src_dir, use_sas=False):
    visits = visit_to_month["oai"]
    dataset_name = "oai"
    visits = ['00', '12', '24', '36', '48', '72', '96']
    exam_codes = ['00', '01', '03', '05', '06', '08', '10']
    rows = []
    sides = [None, 'R', 'L']
    for i, visit in enumerate(visits):
        print(f'==> Reading OAI {visit} visit')
        if use_sas:
            meta = read_sas7bdata_pd(os.path.join(oai_src_dir,
                                                  'Semi-Quant Scoring_SAS',
                                                  f'kxr_sq_bu{exam_codes[i]}.sas7bdat'))
        else:
            meta = pd.read_csv(os.path.join(oai_src_dir,
                                            'Semi-Quant Scoring_ASCII',
                                            f'kxr_sq_bu{exam_codes[i]}.txt'), sep='|')

        meta['ID'] = meta['ID'].astype(str)
        meta.replace({'SIDE': {1: sides[1], 2: sides[2]}}, inplace=True)

        meta['visit'] = visits[i]
        meta['visit_code'] = exam_codes[i]
        meta['visit_id'] = int(visits[i]) // 12

        data_clinical = build_clinical_oai(
            oai_src_dir=oai_src_dir, use_sas=use_sas, visit_code=exam_codes[i])
        data_clinical['ID'] = data_clinical['ID'].astype(str)
        data_clinical.rename(columns={'Side': 'SIDE'}, inplace=True)
        meta = meta.merge(data_clinical, on=[
                          'ID', 'SIDE', 'visit_id', 'visit', 'visit_code'], how='left')
        meta.replace({'SEX': {'1: Male': 0, '2: Female': 1}}, inplace=True)

        # Dropping the data from multiple projects
        meta.drop_duplicates(subset=['ID', 'SIDE'], inplace=True)
        meta.fillna(-1, inplace=True)
        for c in meta.columns:
            meta[c.upper()] = meta[c]

        meta.rename(columns={'SIDE': 'Side', 'V00SITE': 'SITE'}, inplace=True)

        meta['KL'] = meta[f'V{exam_codes[i]}XRKL']

        meta['OSTL'] = meta[f'V{exam_codes[i]}XROSTL']
        meta['OSTM'] = meta[f'V{exam_codes[i]}XROSTM']

        meta['OSFL'] = meta[f'V{exam_codes[i]}XROSFL']
        meta['OSFM'] = meta[f'V{exam_codes[i]}XROSFM']

        meta['JSL'] = meta[f'V{exam_codes[i]}XRJSL']
        meta['JSM'] = meta[f'V{exam_codes[i]}XRJSM']

        meta = meta[['ID', 'Side', 'KL', 'visit', 'visit_id', 'visit_code', 'OSTL', 'OSTM', 'OSFL', 'OSFM', 'JSL', 'JSM',
                     'AGE', 'SEX', 'BMI', 'INJ', 'SURG', 'WOMAC', 'SITE']]

        for index, row in tqdm(meta.iterrows(), total=len(meta.index), desc="Loading OAI meta"):
            img_fullname = os.path.join(
                img_dir, row['ID'] + "_" + row['visit'] + "_" + row['Side'] + ".png")
            if os.path.isfile(img_fullname):
                rows.append(row)
            else:
                print(f'Not found {img_fullname}')

    return pd.DataFrame(rows, index=None)


def build_img_klg_meta_most(img_dir, most_src_dir):
    dataset_name = "most"
    data = read_sas7bdata_pd(os.path.join(
        most_src_dir, 'mostv01235xray.sas7bdat')).fillna(-1)
    data.set_index('MOSTID', inplace=True)
    rows = []
    # Assumption: Only use visit 0 (baseline)
    files = glob.glob(os.path.join(most_src_dir, '*enroll.sas7bdat'))
    files_dict = {file.split('/')[-1].lower(): file for file in files}

    enrolled = {}
    meta_clinical = []
    for visit_id, visit in enumerate([0, 1, 2, 3, 5]):
        print(f'==> Reading MOST {visit} visit')
        ds = read_sas7bdata_pd(files_dict[f'mostv{visit}enroll.sas7bdat'])
        if 'V{}PA'.format(visit) in ds:
            # Filtering out the cases when X-rays wern't taken
            ds = ds[ds['V{}PA'.format(visit)] == 1]
        id_set = set(ds.MOSTID.values.tolist())
        enrolled[visit] = id_set
        meta_clinical.append(build_clinical_most(
            most_src_dir=most_src_dir, visit_code=visit, visit_id=visit_id))

    meta_clinical = pd.concat(meta_clinical, axis=0)

    rows = []
    for i, visit in enumerate([0, 1, 2, 3, 5]):
        for ID in tqdm(enrolled[visit], total=len(enrolled[visit]), desc="Loading MOST meta"):
            subj = data.loc[ID]
            kl_l_key = 'V{0}X{1}{2}'.format(visit, 'L', 'KL')
            kl_r_key = 'V{0}X{1}{2}'.format(visit, 'R', 'KL')
            if kl_l_key in subj and kl_r_key in subj:
                KL_bl_l = subj[kl_l_key]
                KL_bl_r = subj[kl_r_key]
                fullname_l = os.path.join(
                    img_dir, ID + "_" + visit_to_month["most"][i] + "_L.png")
                fullname_r = os.path.join(
                    img_dir, ID + "_" + visit_to_month["most"][i] + "_R.png")

                if os.path.isfile(fullname_l):
                    rows.append({'ID': ID, 'Side': 'L', 'KL': KL_bl_l, 'visit_id': i, 'visit': visit_to_month["most"][i],
                                'dataset': dataset_name})
                else:
                    print(f'Not found {fullname_l}')

                if os.path.isfile(fullname_r):
                    rows.append({'ID': ID, 'Side': 'R', 'KL': KL_bl_r, 'visit_id': i, 'visit': visit_to_month["most"][i],
                                'dataset': dataset_name})
                else:
                    print(f'Not found {fullname_r}')

    data = pd.DataFrame(
        rows, columns=['ID', 'Side', 'KL', 'visit_id', 'visit', 'dataset'])

    data = pd.merge(data, meta_clinical, how="left", on=[
                    'ID', 'Side', 'visit_id', 'visit'])

    return data


def get_most_meta(meta_path):
    # SIDES numbering is made according to the OAI notation
    # SIDE=1 - Right
    # SIDE=2 - Left
    print('==> Processing', os.path.join(meta_path, 'mostv01235xray.sas7bdat'))
    most_meta = read_sas7bdata_pd(os.path.join(
        meta_path, 'mostv01235xray.sas7bdat'))

    most_names_list = pd.read_csv(os.path.join(
        meta_path, 'MOST_names.csv'), header=None)[0].values.tolist()
    xray_types = pd.DataFrame(
        list(map(lambda x: (x.split('/')[0][:-5], follow_up_dict_most[int(x.split('/')[1][1])], x.split('/')[-2]),
                 most_names_list)), columns=['ID', 'visit', 'TYPE'])

    most_meta_all = []
    for visit_id in [0, 1, 2, 3, 5]:
        for leg in ['L', 'R']:
            features = ['MOSTID', ]
            for compartment in ['L', 'M']:
                for bone in ['F', 'T']:
                    features.append(f"V{visit_id}X{leg}OS{bone}{compartment}"),
                features.append(f"V{visit_id}X{leg}JS{compartment}")
            features.append(f"V{visit_id}X{leg}KL")
            tmp = most_meta.copy()[features]
            trunc_feature_names = list(
                map(lambda x: 'XR' + x[4:], features[1:]))
            tmp[trunc_feature_names] = tmp[features[1:]]
            tmp.drop(features[1:], axis=1, inplace=True)
            tmp['Side'] = leg  # int(1 if leg == 'R' else 2)
            tmp = tmp[~tmp.isnull().any(1)]
            tmp['visit'] = follow_up_dict_most[visit_id]
            tmp['ID'] = tmp['MOSTID'].copy()
            tmp.drop('MOSTID', axis=1, inplace=True)
            most_meta_all.append(tmp)

    most_meta = pd.concat(most_meta_all)
    most_meta = most_meta[(most_meta[trunc_feature_names] <= 4).all(1)]
    most_meta = pd.merge(xray_types, most_meta)
    most_meta = most_meta[most_meta.TYPE == 'PA10']
    most_meta.drop('TYPE', axis=1, inplace=True)
    return most_meta


def filter_most_by_pa(ds, df_most_ex, pas=['PA05', 'PA10', 'PA15']):
    std_rows = []
    for i, row in df_most_ex.iterrows():
        std_row = dict()
        std_row['ID'] = row['ID_ex'].split('_')[0]
        std_row['visit_id'] = int(row['visit'][1:])
        std_row['PA'] = row['PA']
        std_rows.append(std_row)
    df_most_pa = pd.DataFrame(std_rows)

    ds_most_filtered = pd.merge(ds, df_most_pa, on=['ID', 'visit_id'])
    if isinstance(pas, str):
        ds_most_filtered = ds_most_filtered[ds_most_filtered['PA'] == pas]
    return ds_most_filtered


def build_clinical_oai(oai_src_dir, use_sas=False, visit_code='00', visit_id=0):
    if use_sas:
        data_enrollees = read_sas7bdata_pd(os.path.join(
            oai_src_dir, 'AllClinical_SAS', 'enrollees.sas7bdat'))
        data_clinical = read_sas7bdata_pd(
            os.path.join(oai_src_dir, 'AllClinical_SAS', f'allclinical{visit_code}.sas7bdat'))
    else:
        data_enrollees = pd.read_csv(os.path.join(
            oai_src_dir, 'General_ASCII', 'Enrollees.txt'), sep='|')
        data_clinical = pd.read_csv(os.path.join(oai_src_dir, 'AllClinical_ASCII', f'AllClinical{visit_code}.txt'),
                                    sep='|')

    clinical_data_oai = data_clinical.merge(data_enrollees, on='ID')

    if not use_sas:
        clinical_data_oai = clinical_data_oai.replace(
            {'0: No': 0, '1: Yes': 1})

    if visit_code == '00':
        AGE_col = 'V00AGE'
        BMI_col = 'P01BMI'
        HEIGHT_col = 'P01HEIGHT'
        WEIGHT_col = 'P01WEIGHT'
        INJL_col = 'P01INJL'
        INJR_col = 'P01INJR'
        SURGL_col = 'P01KSURGL'
        SURGR_col = 'P01KSURGR'
        WOMACL_col = 'V00WOMTSL'
        WOMACR_col = 'V00WOMTSR'
    else:
        AGE_col = f'V{visit_code}AGE'
        BMI_col = f'V{visit_code}BMI'
        HEIGHT_col = f'V{visit_code}HEIGHT'
        WEIGHT_col = f'V{visit_code}WEIGHT'
        INJL_col = f'V{visit_code}INJL12'
        INJR_col = f'V{visit_code}INJR12'
        SURGL_col = f'V{visit_code}KSRGL12'
        SURGR_col = f'V{visit_code}KSRGR12'
        WOMACL_col = f'V{visit_code}WOMTSL'
        WOMACR_col = f'V{visit_code}WOMTSR'

    # Age, Sex, BMI
    clinical_data_oai['SEX'] = clinical_data_oai['P02SEX']
    clinical_data_oai['AGE'] = clinical_data_oai[AGE_col]
    clinical_data_oai['BMI'] = clinical_data_oai[BMI_col]

    # clinical_data_oai['HEIGHT'] = clinical_data_oai[HEIGHT_col]
    # clinical_data_oai['WEIGHT'] = clinical_data_oai[WEIGHT_col]

    clinical_data_oai_left = clinical_data_oai.copy()
    clinical_data_oai_right = clinical_data_oai.copy()

    # Making side-wise metadata
    clinical_data_oai_left['Side'] = 'L'
    clinical_data_oai_right['Side'] = 'R'

    # Injury (ever had)
    clinical_data_oai_left['INJ'] = clinical_data_oai_left[INJL_col]
    clinical_data_oai_right['INJ'] = clinical_data_oai_right[INJR_col]

    # Surgery (ever had)
    clinical_data_oai_left['SURG'] = clinical_data_oai_left[SURGL_col]
    clinical_data_oai_right['SURG'] = clinical_data_oai_right[SURGR_col]

    # Total WOMAC score
    clinical_data_oai_left['WOMAC'] = clinical_data_oai_left[WOMACL_col]
    clinical_data_oai_right['WOMAC'] = clinical_data_oai_right[WOMACR_col]

    clinical_data_oai_left['V00SITE'] = clinical_data_oai['V00SITE']
    clinical_data_oai_right['V00SITE'] = clinical_data_oai['V00SITE']

    clinical_data_oai = pd.concat(
        (clinical_data_oai_left, clinical_data_oai_right))
    clinical_data_oai.ID = clinical_data_oai.ID.values.astype(str)

    clinical_data_oai['visit'] = visit_to_month['oai'][visit_id]
    clinical_data_oai['visit_id'] = visit_id
    clinical_data_oai['visit_code'] = visit_code

    for col in ['BMI', 'INJ', 'SURG', 'WOMAC']:
        clinical_data_oai.loc[clinical_data_oai[col].isin(
            ['.: Missing Form/Incomplete Workbook'])] = None
        clinical_data_oai.loc[clinical_data_oai[col] < 0, col] = None
    return clinical_data_oai[['ID', 'Side', 'visit', 'visit_id', 'visit_code', 'AGE', 'SEX', 'BMI', 'INJ', 'SURG', 'WOMAC', 'V00SITE']]


def build_clinical_most(most_src_dir, visit_code=0, visit_id=0):
    files = glob.glob(os.path.join(most_src_dir, '*enroll.sas7bdat'))
    files_dict = {file.split('/')[-1].lower(): file for file in files}
    clinical_data_most = read_sas7bdata_pd(
        files_dict[f'mostv{visit_code}enroll.sas7bdat'])
    clinical_data_most['ID'] = clinical_data_most.MOSTID
    clinical_data_most['BMI'] = clinical_data_most[f'V{visit_code}BMI']

    clinical_data_most_left = clinical_data_most.copy()
    clinical_data_most_right = clinical_data_most.copy()

    # Making side-wise metadata
    clinical_data_most_left['Side'] = 'L'
    clinical_data_most_right['Side'] = 'R'

    # Injury (ever had)
    clinical_data_most_left['INJ'] = clinical_data_most_left[f'V{visit_code}LAL']
    clinical_data_most_right['INJ'] = clinical_data_most_right[f'V{visit_code}LAR']

    # Surgery (ever had)
    clinical_data_most_left['SURG'] = clinical_data_most_left[f'V{visit_code}SURGL']
    clinical_data_most_right['SURG'] = clinical_data_most_right[f'V{visit_code}SURGR']

    # Total WOMAC score
    clinical_data_most_left['WOMAC'] = clinical_data_most_left[f'V{visit_code}WOTOTL']
    clinical_data_most_right['WOMAC'] = clinical_data_most_right[f'V{visit_code}WOTOTR']

    # Visit
    clinical_data_most_left['visit_id'] = visit_code
    clinical_data_most_right['visit_id'] = visit_code

    clinical_data_most_left['visit'] = visit_to_month["most"][visit_id]
    clinical_data_most_right['visit'] = visit_to_month["most"][visit_id]

    clinical_data_most = pd.concat(
        (clinical_data_most_left, clinical_data_most_right))

    if visit_id == 0:
        return clinical_data_most[['ID', 'Side', 'visit', 'visit_id', 'AGE', 'SEX', 'BMI', 'INJ', 'SURG', 'WOMAC', 'SITE']]
    else:
        return clinical_data_most[['ID', 'Side', 'visit', 'visit_id', 'SEX', 'BMI', 'INJ', 'SURG', 'WOMAC', 'SITE']]


def load_most_metadata(root, save_dir, force_reload=False):
    most_meta_fullname = os.path.join(save_dir, most_meta_csv_filename)
    most_participants_fullname = os.path.join(
        save_dir, most_participants_csv_filename)
    most_all_fullname = os.path.join(save_dir, oai_most_all_csv_filename)

    requires_update = False

    if os.path.isfile(most_meta_fullname) and not force_reload:
        most_meta = pd.read_csv(most_meta_fullname, sep='|')
    else:
        most_meta = build_img_klg_meta_most(os.path.join(root, 'most_meta/'))
        most_meta_strict = get_most_meta(os.path.join(root, 'most_meta/'))

        most_meta = pd.merge(most_meta, most_meta_strict,
                             on=('ID', 'Side', 'visit'), how='inner')
        most_meta.to_csv(most_meta_fullname, index=None, sep='|')
        requires_update = True

    if os.path.isfile(most_participants_fullname) and not force_reload:
        most_ppl = pd.read_csv(most_participants_fullname, sep='|')
    else:
        most_ppl = build_clinical_most(os.path.join(root, 'most_meta/'))
        most_ppl.to_csv(most_participants_fullname, index=None, sep='|')
        requires_update = True

    master_dict = {"oai": dict(), "most": dict()}
    master_dict["most"]["meta"] = most_meta
    master_dict["most"]["ppl"] = most_ppl

    master_dict["most"]["n_dup"] = dict()

    master_dict["most"]["n_dup"]["meta"] = len(
        most_meta[most_meta.duplicated(keep=False)].index)
    master_dict["most"]["n_dup"]["ppl"] = len(
        most_ppl[most_ppl.duplicated(keep=False)].index)

    for ds in ["most"]:
        for s in ["meta", "ppl"]:
            if master_dict[ds]["n_dup"][s] > 0:
                print(master_dict[ds][s])
                raise ValueError(
                    "There are {} duplicated rows in {} {} dataframe".format(master_dict[ds]["n_dup"][s], ds.upper(),
                                                                             s.upper()))

    master_dict["most"]["all"] = pd.merge(master_dict["most"]["meta"],
                                          master_dict["most"]["ppl"], how="left",
                                          left_on=["ID", "Side"], right_on=["ID", "Side"]).fillna(-1)

    return master_dict


def load_oai_most_metadata(img_root, metadata_root, save_dir, force_reload=False):
    oai_meta_fullname = os.path.join(save_dir, oai_meta_csv_filename)
    oai_participants_fullname = os.path.join(
        save_dir, oai_participants_csv_filename)
    most_meta_fullname = os.path.join(save_dir, most_meta_csv_filename)
    most_participants_fullname = os.path.join(
        save_dir, most_participants_csv_filename)
    oai_most_meta_fullname = os.path.join(save_dir, oai_most_meta_csv_filename)
    oai_most_all_fullname = os.path.join(save_dir, oai_most_all_csv_filename)

    if os.path.isfile(oai_meta_fullname) and not force_reload:
        oai_meta = pd.read_csv(oai_meta_fullname, sep=',')
        oai_meta["ID"] = oai_meta["ID"].values.astype(str)
    else:
        oai_meta = build_img_klg_meta_oai(img_dir=img_root, oai_src_dir=os.path.join(
            metadata_root, 'X-Ray_Image_Assessments_SAS/'))
        print(f'Saving OAI metadata to {oai_meta_fullname}')
        oai_meta.to_csv(oai_meta_fullname, index=None, sep=',')

    if os.path.isfile(most_meta_fullname) and not force_reload:
        most_meta = pd.read_csv(most_meta_fullname, sep='|')
    else:
        most_meta = build_img_klg_meta_most(
            img_dir=img_root, most_src_dir=os.path.join(metadata_root, 'most_meta/'))
        print(f'Saving MOST metadata to {most_meta_fullname}')
        most_meta.to_csv(most_meta_fullname, index=None, sep=',')


def crop_2_rois_oai_most(img, ps=128, debug=False):
    """
    Generates pair of images 128x128 from the knee joint.
    ps shows how big area should be mapped into that region.
    """
    if len(img.shape) > 2:
        if img.shape[0] == 1:
            img = np.squeeze(img, axis=0)
        else:
            raise ValueError('Input image ({}) must have 1 channel, but got {}'.format(
                img.shape, img.shape[0]))
    elif len(img.shape) < 2:
        raise ValueError(
            'Input image ({}) must have 2 dims or 1-channel 3 dims, but got {}'.format(img.shape, img.shape))
    elif img.shape[0] < 300 or img.shape[1] < 300:
        raise ValueError('Input image shape ({}) must be at least 300, but got {}'.format(
            img.shape, img.shape))

    s = img.shape[0]

    s1, s2, pad = calc_roi_bboxes(s, ps)

    # pad = int(np.floor(s / 3))
    # l = img[pad:pad+ps, 0:ps]
    # m = img[pad:pad+ps, s-ps:s]

    l = img[s1[0]:s1[2], s1[1]:s1[3]]
    m = img[s2[0]:s2[2], s2[1]:s2[3]]

    # DEBUG
    if debug:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(img, (0, pad), (ps, pad + ps),
                      color=(255, 255, 0), thickness=4)
        cv2.rectangle(img, (s - ps, pad), (s, pad + ps),
                      color=(255, 255, 0), thickness=4)
        return l, m, img
    else:
        return l, m, None


def calc_roi_bboxes(s, ps):
    pad = int(np.floor(s / 3))

    s1 = (pad, 0, pad + ps, ps)
    s2 = (pad, s - ps, pad + ps, s)
    return s1, s2, pad


def standardize_img(img, std_actual_shape=(130, 130), target_shape=(300, 300),
                    original_actual_shape=(140, 140), original_img_shape=(700, 700)):
    spacing_per_pixel = (1.0 * original_actual_shape[0] / original_img_shape[0],
                         1.0 * original_actual_shape[1] / original_img_shape[1])
    std_img_shape = (int(std_actual_shape[0] / spacing_per_pixel[0]),
                     int(std_actual_shape[1] / spacing_per_pixel[1]))
    cropped_img = center_crop(img, std_img_shape)
    cropped_img = cv2.resize(
        cropped_img, dsize=target_shape, interpolation=cv2.INTER_AREA)
    return cropped_img


def overlay_heatmap(heatmap, mask, whole_img, box, blend_w=0.7, std_actual_shape=(110, 110), target_shape=(300, 300),
                    original_actual_shape=(140, 140), original_img_shape=(700, 700), draw_crops=False,
                    crop_center=False):
    spacing_per_pixel = (1.0 * original_actual_shape[0] / original_img_shape[0],
                         1.0 * original_actual_shape[1] / original_img_shape[1])
    std_img_shape = (int(std_actual_shape[0] / spacing_per_pixel[0]),
                     int(std_actual_shape[1] / spacing_per_pixel[1]))

    s = std_img_shape[0] / target_shape[0]

    heatmap = cv2.resize(heatmap, dsize=(0, 0), fx=s,
                         fy=s, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, dsize=(0, 0), fx=s, fy=s,
                      interpolation=cv2.INTER_AREA)

    box = list(box)

    for i in range(len(box)):
        box[i] *= s

    c_y = int(round((whole_img.shape[0] - std_img_shape[0]) / 2))
    c_x = int(round((whole_img.shape[1] - std_img_shape[1]) / 2))

    box[0] += c_y
    box[1] += c_x
    box[2] += c_y
    box[3] += c_x

    if whole_img.shape[2] == 1:
        whole_img = cv2.cvtColor(whole_img, cv2.COLOR_GRAY2BGR)

    box = [int(round(box[i])) for i in range(len(box))]

    # print(f'box shape = ({box[2]-box[0]}, {box[3]-box[1]})')

    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, axis=-1)

    if len(heatmap.shape) == 2:
        heatmap = np.expand_dims(heatmap, axis=-1)

    whole_img = np.float32(whole_img) / 255
    blend_whole_img = blend_heatmap(whole_img, heatmap, mask, blend_w, box)

    if draw_crops:
        cv2.rectangle(whole_img, (box[1], box[0]),
                      (box[3], box[2]), (0, 123, 247), 4)
    # plt.imshow(whole_img)
    # plt.show()

    # plt.imshow(blend_whole_img)
    # plt.show()
    #
    # plt.imshow(heatmap)
    # plt.show()

    blend_whole_img = np.uint8(255 * blend_whole_img)

    return blend_whole_img, heatmap, mask, box


def blend_heatmap(img, heatmap, mask, w=0.8, box=None):
    if box is None:
        img = 1 * (1 - mask ** w) * img + (mask ** w) * heatmap
    elif (isinstance(box, tuple) or isinstance(box, list)) and len(box) == 4:
        img[box[0]:box[2], box[1]:box[3], :] = 1 * (1 - mask ** w) * img[box[0]:box[2], box[1]:box[3], :] + (
            mask ** w) * heatmap
    else:
        raise ValueError(f'Invalid input box with type of {type(box)}')

    return img


def center_crop(img, crop_sz):
    c_y = int(round((img.shape[0] - crop_sz[0]) / 2))
    c_x = int(round((img.shape[1] - crop_sz[1]) / 2))
    cropped_img = img[c_y:c_y + crop_sz[0], c_x:c_x + crop_sz[1]]
    return cropped_img


def crop_img2(img, crop_side_ratio=0.55):
    # Assumption: Input image is square
    sz = round(img.shape[0] * crop_side_ratio)
    center = (img.shape[0] // 2, img.shape[1] // 2)

    tl1 = (center[0] - sz // 2, 0)
    br1 = (tl1[0] + sz, tl1[1] + sz)

    br2 = (br1[0], img.shape[1] - 1)
    tl2 = (br2[0] - sz, br2[1] - sz)

    img1 = img[tl1[0]:br1[0], tl1[1]:br1[1]]
    img2 = img[tl2[0]:br2[0], tl2[1]:br2[1]]

    return img1, img2


def load_oai_most_datasets(root, img_dir, save_meta_dir, saved_patch_dir, output_filename, force_reload=False,
                           force_rewrite=False, extract_sides=True):
    if not os.path.exists(save_meta_dir):
        os.mkdir(save_meta_dir)

    if not os.path.exists(saved_patch_dir):
        os.mkdir(saved_patch_dir)

    if force_rewrite or not os.path.exists(os.path.join(save_meta_dir, output_filename)):
        print('Loading OAI MOST metadata...')
        df_meta = load_oai_most_metadata(
            metadata_root=root, save_dir=save_meta_dir, force_reload=force_reload)
        print('OK!')
        df = df_meta["oai_most"]["all"]
        fullnames = []
        for index, row in tqdm(df.iterrows(), total=len(df.index), desc="Processing images"):
            fname = row["ID"] + "_" + visit_to_month[row["dataset"]
                                                     ][row["visit_id"]] + "_" + row["Side"] + ".png"
            img_fullname = os.path.join(img_dir, fname)
            basename = os.path.splitext(fname)[0]

            img1_fullname = os.path.join(
                saved_patch_dir, "{}_patch1.png".format(basename))
            img2_fullname = os.path.join(
                saved_patch_dir, "{}_patch2.png".format(basename))

            img_cropped_fullname = os.path.join(
                saved_patch_dir, "{}_cropped.png".format(basename))

            fullnames_dict = row.to_dict()
            fullnames_dict['Filename'] = None
            fullnames_dict['Patch1_name'] = None
            fullnames_dict['Patch2_name'] = None

            fullnames_dict['ID'] = row['ID']
            fullnames_dict['Side'] = row['Side']
            if os.path.exists(img_fullname) and row["KL"] > -1 and row["KL"] < 5:
                if (not os.path.isfile(img_cropped_fullname) and not extract_sides) or \
                        ((not os.path.isfile(img1_fullname) or not os.path.isfile(img2_fullname)) and extract_sides):
                    img = cv2.imread(img_fullname, cv2.IMREAD_GRAYSCALE)

                    img = standardize_img(img, std_actual_shape=(
                        110, 110), target_shape=(300, 300))
                    if extract_sides:
                        img1, img2, _ = crop_2_rois_oai_most(img)
                        cv2.imwrite(img1_fullname, img1)
                        cv2.imwrite(img2_fullname, img2)
                    else:
                        cv2.imwrite(img_cropped_fullname, img)

                fullnames_dict['Filename'] = fname
                if extract_sides:
                    fullnames_dict['Patch1_name'] = os.path.basename(
                        img1_fullname)
                    fullnames_dict['Patch2_name'] = os.path.basename(
                        img2_fullname)
                else:
                    fullnames_dict['ROI_name'] = os.path.basename(
                        img_cropped_fullname)
                fullnames.append(fullnames_dict)

        df_all = pd.DataFrame(fullnames, index=None)
        df_all.to_csv(os.path.join(
            save_meta_dir, output_filename), index=None, sep='|')
    else:
        df_all = pd.read_csv(os.path.join(
            save_meta_dir, output_filename), sep='|')
    return df_all


def load_most_dataset(root, img_dir, save_meta_dir, saved_patch_dir, output_filename, force_reload=False,
                      force_rewrite=False):
    if not os.path.exists(save_meta_dir):
        os.mkdir(save_meta_dir)

    if not os.path.exists(saved_patch_dir):
        os.mkdir(saved_patch_dir)

    if force_rewrite or not os.path.exists(os.path.join(save_meta_dir, output_filename)):
        df_meta = load_most_metadata(
            root=root, save_dir=save_meta_dir, force_reload=force_reload)
        df = df_meta["most"]["all"]
        fullnames = []
        for index, row in tqdm(df.iterrows(), total=len(df.index), desc="Processing images"):
            fname = row["ID"] + "_" + visit_to_month[row["dataset"]
                                                     ][row["visit_id"]] + "_" + row["Side"] + ".png"
            img_fullname = os.path.join(img_dir, fname)
            basename = os.path.splitext(fname)[0]
            img1_fullname = os.path.join(
                saved_patch_dir, "{}_patch1.png".format(basename))
            img2_fullname = os.path.join(
                saved_patch_dir, "{}_patch2.png".format(basename))
            fullnames_dict = row.to_dict()
            fullnames_dict['Filename'] = None
            fullnames_dict['Patch1_name'] = None
            fullnames_dict['Patch2_name'] = None

            fullnames_dict['ID'] = row['ID']
            fullnames_dict['Side'] = row['Side']
            if not os.path.exists(img_fullname):
                print('Not found file {}'.format(img_fullname))
            elif row["KL"] < 0 and row["KL"] > 4:
                print('KL {} is out of range'.format(row["KL"]))
            else:
                if not os.path.isfile(img1_fullname) or not os.path.isfile(img2_fullname):
                    img = cv2.imread(img_fullname, cv2.IMREAD_GRAYSCALE)

                    img = standardize_img(img, std_actual_shape=(
                        110, 110), target_shape=(300, 300))
                    img1, img2, _ = crop_2_rois_oai_most(img)
                    cv2.imwrite(img1_fullname, img1)
                    cv2.imwrite(img2_fullname, img2)

                fullnames_dict['Filename'] = fname
                fullnames_dict['Patch1_name'] = os.path.basename(img1_fullname)
                fullnames_dict['Patch2_name'] = os.path.basename(img2_fullname)
                fullnames.append(fullnames_dict)

        df_all = pd.DataFrame(fullnames, index=None)
        print('Save {} lines into {}'.format(len(df_all),
              os.path.join(save_meta_dir, output_filename)))
        df_all.to_csv(os.path.join(
            save_meta_dir, output_filename), index=None, sep='|')
    else:
        df_all = pd.read_csv(os.path.join(
            save_meta_dir, output_filename), sep='|')
    return df_all
