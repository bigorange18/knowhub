'''
Author: chenorange chenorange2219@gmail.com
Date: 2025-05-16 07:54:36
LastEditors: chenorange chenorange2219@gmail.com
LastEditTime: 2025-05-16 07:57:06
FilePath: \knowhub\pytest\example\test.py
Description: MLPredictor 的 pytest 用例（随机森林 + iris）

Copyright (c) 2025 by ${git_name_email}, All Rights Reserved.
'''
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import MLPredictor  # noqa: E402

SAMPLE = [5.1, 3.5, 1.4, 0.2]


@pytest.fixture()
def predictor():
    p = MLPredictor()
    p.train()
    return p


def test_train(predictor):
    assert predictor.model is not None


def test_predict(predictor):
    assert len(predictor.predict(SAMPLE)) == 1


def test_predict_before_train():
    with pytest.raises(Exception, match="not trained"):
        MLPredictor().predict(SAMPLE)


def test_save_and_load(predictor, tmpdir):
    path = str(tmpdir.join("model.pkl"))
    predictor.save_model(path)

    loaded = MLPredictor()
    loaded.load_model(path)
    assert loaded.predict(SAMPLE)[0] == predictor.predict(SAMPLE)[0]
