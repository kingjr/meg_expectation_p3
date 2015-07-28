# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# 1. XXX check that SOA is nan if absent, and != from soa_ttl
# 2. Find better systematic names
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
from sklearn.linear_model import LogisticRegression

# Soa only defined for target-present trials.
# Need to use soa_ttl to include absent.

soas = [17, 33, 50, 67, 83]
mid_soas = [33, 50, 67]


class clf_2class_proba(LogisticRegression):
    """Probabilistic SVC for 2 classes only"""
    def predict(self, x):
        probas = super(clf_2class_proba, self).predict_proba(x)
        return probas[:, 1]


# Key contrasts for replication
def analysis(name, typ, condition=None, query=None):
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVR
    from toolbox.jr_toolbox.utils import (scorer_auc, scorer_spearman)
    single_trial = False
    erf_function = None
    if typ == 'categorize':
        clf = Pipeline([('scaler', StandardScaler()),
                        ('svc', clf_2class_proba(C=1, class_weight='auto'))])
        scorer = scorer_auc
        chance = .5
    elif typ == 'regress':
        clf = Pipeline([('scaler', StandardScaler()), ('svr', LinearSVR(C=1))])
        scorer = scorer_spearman
        single_trial = True  # with non param need single trial here
        chance = 0.
    if condition is None:
        condition = name
    return dict(name=name, condition=condition, query=query, clf=clf,
                scorer=scorer, chance=chance, function=erf_function,
                single_trial=single_trial, cv=8)

# Present vs. absent
contrast_pst = analysis('presence', 'categorize', condition='present')

# PAS for all trials
contrast_pst = analysis('pas_all', 'regress', condition='pas')

# PAS for present trials only
regress_pas_pst = analysis('pas_pst', 'regress', condition='pas',
                           query='present==True')

# PAS for present middle SOA trials only
regress_pas_mid = analysis('pas_pst_mid', 'regress', condition='pas',
                           query='present==True and soa!=17 and soa!=83')

# Seen vs. unseen for all trials
contrast_seen_all = analysis('seen_all', 'categorize', condition='seen')

# Seen vs. unseen for present trials only
contrast_seen_pst = analysis('seen_pst', 'categorize', condition='seen',
                             query='present==True')

# Seen vs. unseen for present middle SOA trials
contrast_seen_pst_mid = analysis('seen_pst_mid', 'categorize', condition='seen',
                                 query='present==True and soa!=17 and soa!=83')

# Absent, unseen present, seen present
regress_abs_seen = analysis('abs_seen', 'regress', condition='abs_seen')

# 5 SOAs compared to absent
regress_abs_seen = analysis('abs_soa', 'regress', condition='abs_soa')

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# XXX check that SOA is nan if absent, and != from soa_ttl
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
pas_regress_list = list()
for soa in soas:
    sub_regress_ = analysis('soa_%s' % soa, 'regress', condition='pas',
                            query='soa==%s' % soa)
    pas_regress_list.append(sub_regress_)
regress_pas_soa = analysis('soa_pas', 'regress', condition=pas_regress_list)

# Effects of visibility context

# # Global block context (middle SOAs only)
contrast_block = analysis('block','categorize', condition='block',
                          query='present==True and soa!=17 and soa!=83')

# Global block context, SOA (middle SOAs only)
contrast_block_soa = list()
for soa in mid_soas:
    sub_contrast_ = analysis('soa_%s' % soa, 'categorize', condition='block',
                        query='soa==%s' % soa)
    contrast_block_soa.append(sub_contrast_)
regress_block_soa = analysis('block_soa', 'regress',
                             condition=contrast_block_soa)

pas_regress_list = list()
for soa in soas:
    sub_regress_ = analysis('soa_%s' % soa, 'regress', condition='pas',
                            query='soa==%s' % soa)
    pas_regress_list.append(sub_regress_)
regress_pas_soa = analysis('soa_pas', 'regress', condition=pas_regress_list)


# # Global block context, PAS (middle SOA, present only)
block_contrast_list = list()
for pas in range(4):
    sub_contrast_ = analysis('pas_%s' % pas, 'categorize', condition='block',
                             query='pas==pas and present==True and soa!=17 and soa!=83')
    block_contrast_list.append(sub_contrast_)
regress_block_pas = analysis('block_pas', 'regress',
                             condition = block_contrast_list)

# # Global block context, seen/unseen (present, middle SOAs only)
block_contrast_list = list()
for block in ['visible', 'invisible']:
    sub_contrast_ = analysis('seen_'+block[:-4], 'categorize', condition='seen',
                             query='present==True and soa!=17 and soa!=83 and block=="%s"' % block)
    block_contrast_list.append(sub_contrast_)
contrast_block_seen = analysis('block_seen','categorize',
                               condition = block_contrast_list)

# Local N-1 context (all trials)
contrast_local = analysis('local','categorize', condition = 'local_context')


# # Local N-1 context, SOA (all trials)
# contrast_local_soa = list()
# for soa in soas:
#     contrast = dict(
#         name='localS'+str(soa)+'-localU'+str(soa),
#         operator = evoked_subtract, conditions = [
#             dict(name='localS'+str(soa), include=dict(local_context='S',soa_ttl=soa)),
#             dict(name='localU'+str(soa), include=dict(local_context='U',soa_ttl=soa))])
#     contrast_local_soa.append(contrast)
# regress_local_soa = dict(
#     name='regress_local_soa',operator=evoked_spearman,
#     conditions=contrast_local_soa)





# # Local N-1 context, PAS (all trials)
# contrast_local_pas = list()
# for pas in range(4):
#     contrast = dict(
#         name='localSpas'+str(pas) + '-localUpas'+str(pas),
#         operator = evoked_subtract, conditions = [
#             dict(name='localSpas'+str(pas), include=dict(local_context='S', pas=pas)),
#             dict(name='localUpas'+str(pas), include=dict(local_context='U', pas=pas))])
#     contrast_local_pas.append(contrast)
# regress_local_pas = dict(
#     name='regress_local_pas', operator=evoked_spearman,
#     conditions=contrast_local_pas)

regress_list = list()
for local in ['S', 'U']:
    regress = analysis('pas_local_%s' % local, 'regress', condition='pas',
                       query='local_context=="%s"' % local)
    regress_list.append(regress)
contrast_local_pas = analysis('contrast_local_pas', 'categorize',
                              condition=regress_list)


# Local N-1 context, seen/unseen (all trials)
contrast_list = list()
for local in ['S', 'U']:
    contrast = analysis('seen_local_%s' % local, 'categorize',
                        condition='seen', query='local_context=="%s"' % local)
    contrast_list.append(contrast)
contrast_local_seen = analysis('contrast_local_seen', 'categorize',
                               condition=contrast_list)

## Control contrasts

# Top vs. bottom: Cannot do because not coded in triggers
contrast_local = analysis('local','categorize', condition = 'local_context')

# Letter vs. digit stimulus (all present trials)
contrast_target = analysis('target','categorize', condition = 'target')

# Motor response (finger to indicate letter or number)
contrast_motor = analysis('motor','categorize', condition = 'letter_resp')


analyses = [contrast_pst, regress_pas_pst, contrast_local_seen]
