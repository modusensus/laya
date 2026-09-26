# Step 1 of v2: verify the installed laya accepts the `labels` key on a noul question
# and see the actual predict() answer structure (do NOT guess key names).
import json, laya

V1 = r'C:\Users\石晴\Desktop\laya-kaggle-output\laya-nli-conflict-v1'
agent = laya.load(V1, device='cpu')

q_new = {'type': 'noul',
         'instructions': '新信息(new)与已有记忆(known)是否冲突？冲突=矛盾需更新旧记忆，兼容=一致或无关',
         'labels': {'false': '兼容', 'true': '冲突'}}
q_old = {'type': 'noul',
         'instructions': '新信息(new)与已有记忆(known)是否冲突（矛盾、需要更新旧记忆）？true=冲突，false=不冲突/互补'}

probes = [
    ({'known': '用户住在上海', 'new': '用户最近搬到北京工作了'}, 'conflict pair'),
    ({'known': '用户住在北京', 'new': '北京有很多历史古迹'}, 'unrelated pair'),
]
for state, note in probes:
    r_new = agent.predict(state, {'conflict': q_new})['answers']['conflict']
    r_old = agent.predict(state, {'conflict': q_old})['answers']['conflict']
    print(f'[{note}]')
    print('  labels-question ->', json.dumps(r_new, ensure_ascii=False))
    print('  v1-style        ->', json.dumps(r_old, ensure_ascii=False))
print('labels key accepted: OK')
