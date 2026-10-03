#!/bin/bash
# Wait for the in-flight E5 to finish pythia-70m, then run E2.
# Sequenced rather than parallel: only ~0.7 GB was free, and an unrelated
# user job (pythia-410m Jacobian) is also resident.
set -u
cd "$(dirname "$0")/.."

echo "waiting for E5 to finish pythia-70m ..."
for i in $(seq 1 240); do
  if python -c "
import json,sys
d=json.load(open('results/data/e5_real_data_multimodel.json'))
sys.exit(0 if not d.get('models_not_completed') else 1)
" 2>/dev/null; then
    echo "E5 complete after ~$((i*15))s"
    break
  fi
  sleep 15
done

python -c "
import json
d=json.load(open('results/data/e5_real_data_multimodel.json'))
print('E5 models:', [m['model'] for m in d['models']], '| partial:', d.get('partial'))
for m in d['models']:
    a=m['aggregate']
    print(f\"  {m['model']:<22} fisher {100*a['fisher_probe']['efficiency_vs_gbar']:5.1f}%  \"
          f\"dom {100*a['diff_of_means']['efficiency_vs_gbar']:5.1f}%  \"
          f\"dark {100*a['dark']['efficiency_vs_gbar']:5.1f}%  reff/d {100*m['mean_r_eff_over_d']:.2f}%\")
"

echo ""
echo "=== E2: trained toy transformers + confound dial ==="
OMP_NUM_THREADS=3 python -u experiments/e2_trained_toy.py 2>&1 | grep -v "it/s" | tail -70
echo "E2 exit=$?"
echo "BOTH_DONE"
