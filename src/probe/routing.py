"""Package a frozen, batch-selected linear head; no inference-time fitting.

All coefficients, class moments, and feature scales are fitted offline.
The only batch-dependent operation chooses one stored head using a fixed
diagonal Gaussian distance. Conditional on that choice, scoring is linear.
"""
from pathlib import Path
import tempfile
import zipfile

import joblib
import numpy as np

from .submission import CLASSIFIER_SOURCE, smoke_test

ROUTER_METHOD = '''    def _select_head(self, X):
        router = self.artifact["router"]
        prior = float(self.artifact["positive_rate"])
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        scale = router["scale"]
        distances = []
        for candidate in router["candidates"]:
            m0, m1 = candidate["mean0"], candidate["mean1"]
            v0, v1 = candidate["var0"], candidate["var1"]
            m = (1-prior)*m0 + prior*m1
            v = (1-prior)*v0 + prior*v1 + prior*(1-prior)*(m1-m0)**2
            s = np.sqrt(np.maximum(v, 1e-12))
            distances.append(np.mean(((mean-m)/scale)**2 + ((std-s)/scale)**2))
        return router["candidates"][int(np.argmin(distances))]["head"]

'''


def classifier_source():
    source = CLASSIFIER_SOURCE
    source = source.replace('    def predict(self, X):', ROUTER_METHOD+'    def predict(self, X):')
    source = source.replace('heads = self.artifact["heads"]', 'heads = [self._select_head(X)]')
    source = source.replace('The class prior is a published property of the task,',
                            'The configured prior is an inference from development aggregates,')
    return source


def package_router(artifact, router, path):
    result = dict(artifact)
    result['version'] = 'corpus_router_with_pooled_fallback'
    result['router'] = router
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        blob = Path(tmp)/'trained_probe.joblib'
        joblib.dump(result, blob, compress=3)
        with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr('classifier.py', classifier_source())
            z.write(blob, 'trained_probe.joblib')
    return {str(n):smoke_test(path, n=n, expect_rate=result['positive_rate']) for n in (1700,1360)}


def candidate(name, head, x, y):
    return {'name':name, 'head':head,
            'mean0':x[y==0].mean(0,dtype=np.float64),
            'mean1':x[y==1].mean(0,dtype=np.float64),
            'var0':x[y==0].var(0,dtype=np.float64),
            'var1':x[y==1].var(0,dtype=np.float64)}
