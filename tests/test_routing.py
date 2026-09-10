"""Behavioral checks for frozen-head selection and serialization."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from probe.routing import classifier_source


def test_router_selects_correct_frozen_head_and_is_stateless():
    namespace = {}
    exec(classifier_source(),namespace)
    c = object.__new__(namespace['Classifier'])
    d = 4
    a={'coef':np.ones(d),'intercept':0,'row_l2':False,'standardize':'none',
       'weight':1,'remove_top':0}
    b={**a,'coef':-np.ones(d)}
    def domain(center,head):
        return {'mean0':np.full(d,center),'mean1':np.full(d,center),
                'var0':np.ones(d),'var1':np.ones(d),'head':head}
    c.artifact={'n_features':d,'positive_rate':.5,'combine':'score','decision':'quota',
                'router':{'scale':np.ones(d),'candidates':[domain(-8,a),domain(8,b)]}}
    x=np.random.default_rng(5).normal(8,1,(40,d))
    assert c._select_head(x) is b
    before=c.predict(x)
    assert np.array_equal(before,(x.sum(1)<=np.median(x.sum(1))).astype(int))
    assert c._select_head(x-16) is a
    c.predict(x-16)
    assert np.array_equal(c.predict(x),before)
    assert c.predict(np.empty((0,d))).shape==(0,)
    order=np.random.default_rng(2).permutation(len(x))
    assert np.array_equal(c.predict(x[order]),before[order])
