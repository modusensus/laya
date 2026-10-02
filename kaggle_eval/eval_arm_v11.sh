#!/bin/bash
# Round-11 per-run local re-verification harness (HANDOFF_NLI_V11.md §1.2/§5).
# usage: eval_arm_v11.sh <ckpt> <tag>      e.g. eval_arm_v11.sh D:/laya-kaggle-output/laya-nli-conflict-v11-r1 r1
CKPT="$1"; TAG="$2"
D=D:/laya/data_local
cd /d/laya/kaggle_eval
P="D:/Miniconda/envs/laya-ft/python.exe"
RC=0
run() { "$@" 2>&1 | tail -3; [ ${PIPESTATUS[0]} -ne 0 ] && RC=1; }

cp "$CKPT/val_probs.json" "$D/val_probs_v11_$TAG.json" || RC=1
echo "== realtest (old20/neg5/new10) [$TAG] =="; run $P realtest_v4.py "$CKPT" "$D/memory_conflict_realtest_v11_$TAG.json"
echo "== dump val_soft [$TAG] =="; run $P dump_probs.py "$CKPT" "$D/val_soft_probs_v11_$TAG.json"
echo "== polarity [$TAG] =="; run $P option_polarity_check.py "$CKPT" "$D/polarity_nli_v11_$TAG.json"
echo "== label swap [$TAG] =="; run $P label_swap_check.py "$CKPT" "$D/label_swap_v11_$TAG.json"
echo "== bias diag [$TAG] =="; run $P noul_bias_diag.py "$CKPT" "$D/noul_bias_diag_v11_$TAG.json"
echo "== band [$TAG] =="; run $P band_report.py --main "$D/val_probs_v11_$TAG.json" --soft "$D/val_soft_probs_v11_$TAG.json" --out "$D/conf_band_v11_$TAG.txt" --round-tag "v11-$TAG"
echo "== in-dist probe [$TAG] =="; run $P probe_indist_v9.py "$CKPT" "$D/indist_probe_v11_$TAG.json"
echo "== conformal alt probe [$TAG] (sel budget 0.34) =="; run $P conformal_alt_probe.py "$CKPT" "$D/conformal_alt_v11_$TAG.json" --val-probs "$D/val_probs_v11_$TAG.json" --sel-budget 0.34
echo "== conformal maxconf archive [$TAG] =="; run $P conformal_abstain.py --main "$D/val_probs_v11_$TAG.json" --soft "$D/val_soft_probs_v11_$TAG.json" --real "$D/memory_conflict_realtest_v11_$TAG.json" --diag "$D/noul_bias_diag_v11_$TAG.json" --out "$D/conformal_v11_$TAG.json" --split-half
echo "== conformal s1 (pinned tie-break) [$TAG] =="; run $P conformal_abstain.py --rule s1_maxconf --split-half --sel-budget 0.34 --alt-scores "$D/conformal_alt_v11_$TAG.json" --main "$D/val_probs_v11_$TAG.json" --soft "$D/val_soft_probs_v11_$TAG.json" --real "$D/memory_conflict_realtest_v11_$TAG.json" --diag "$D/noul_bias_diag_v11_$TAG.json" --out "$D/conformal_v11_${TAG}_s1.json"
echo "== conformal s2 [$TAG] =="; run $P conformal_abstain.py --rule s2 --split-half --sel-budget 0.34 --alt-scores "$D/conformal_alt_v11_$TAG.json" --main "$D/val_probs_v11_$TAG.json" --soft "$D/val_soft_probs_v11_$TAG.json" --real "$D/memory_conflict_realtest_v11_$TAG.json" --diag "$D/noul_bias_diag_v11_$TAG.json" --out "$D/conformal_v11_${TAG}_s2.json"
echo "== family diag [$TAG] (NEW §2.2) =="; run $P family_diag.py "$CKPT" "$D/family_diag_v11_$TAG.json"
echo "== done [$TAG] rc=$RC =="
exit $RC
