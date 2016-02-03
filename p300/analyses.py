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
regress_pas = analysis('pas_all', 'regress', condition='pas',
                       query='pas_undef!=True')

# PAS for present trials only
regress_pas_pst = analysis('pas_pst', 'regress', condition='pas',
                           query='present==True and pas_undef!=True')

# PAS for present middle SOA trials only
regress_pas_mid = analysis('pas_pst_mid', 'regress', condition='pas',
                           query='present==True and soa!=17 and soa!=83 and \
                           pas_undef!=True')

# Seen vs. unseen for all trials
contrast_seen_all = analysis('seen_all', 'categorize', condition='seen',
                             query='seen_undef!=True')

# Seen vs. unseen for present trials only
contrast_seen_pst = analysis('seen_pst', 'categorize', condition='seen',
                             query='present==True and seen_undef!=True')

# Seen vs. unseen for present middle SOA trials
contrast_seen_pst_mid = analysis('seen_pst_mid', 'categorize',
                                 condition='seen', query='present==True and \
                                 seen_undef!=True and soa!=17 and soa!=83')

# Absent, unseen present, seen present
regress_abs_seen = analysis('abs_seen', 'regress', condition='abs_seen',
                            query='seen_undef!=True')

# 5 SOAs compared to absent
regress_abs_soa = analysis('abs_soa', 'regress', condition='abs_soa')

# # Effects of visibility context

# Global block context (present, middle SOAs only)
contrast_block = analysis('block', 'categorize', condition='block',
                          query='present==True and soa!=17 and soa!=83 and \
                          block_undef!=True')

# Global block context, SOA (present, middle SOAs only)
contrast_block_list = list()
for soa in mid_soas:
    sub_contrast_ = analysis('block_soa_%s' % soa, 'categorize',
                             condition='block', query='block_undef!=True \
                             and soa==%s' % soa)
    contrast_block_list.append(sub_contrast_)
regress_block_soa = analysis('block_soa', 'regress',
                             condition=contrast_block_list)

# Global block context, PAS (present, middle SOAs only)
contrast_block_list = list()
for pas in range(4):
    sub_contrast_ = analysis('block_pas_%s' % pas, 'categorize',
                             condition='block', query='block_undef!=True \
                             and present==True and soa!=17 and soa!=83 \
                             and pas==%s' % pas)
    contrast_block_list.append(sub_contrast_)
regress_block_pas = analysis('block_pas', 'regress',
                             condition=contrast_block_list)

# Global block context, seen/unseen (present, middle SOAs only)
contrast_block_list = list()
for seen in ['seen', 'unseen']:
    truth_value = seen == 'seen'
    sub_contrast_ = analysis('block_%s' % seen, 'categorize',
                             condition='block', query='present==True and \
                             soa!=17 and soa!=83 and seen_undef!=True and \
                             seen==%s' % truth_value)
    contrast_block_list.append(sub_contrast_)
contrast_block_seen = analysis('block_seen', 'categorize',
                               condition=contrast_block_list)

# Local N-1 context (all trials)
contrast_local = analysis('local', 'categorize', condition='local_seen',
                          query='local_undef==False and soa!=17 and soa!=83')

# Local N-1 context, SOA (all trials)
contrast_local_list = list()
for soa in soas:
    sub_contrast_ = analysis('local_%s' % soa, 'categorize',
                             condition='local_seen',
                             query='local_undef!=True and \
                                    soa_ttl==%s' % soa)
    contrast_local_list.append(sub_contrast_)
regress_local_soa = analysis('local_soa', 'regress',
                             condition=contrast_local_list)


contrast_local_list = list()
for soa in mid_soas:
    sub_contrast_ = analysis('local_%s_mid' % soa, 'categorize',
                             condition='local_seen',
                             query='local_undef!=True and \
                                    soa_ttl==%s' % soa)
    contrast_local_list.append(sub_contrast_)
regress_local_soa_mid = analysis('local_soa_mid', 'regress',
                                 condition=contrast_local_list)

# Local N-1 context, PAS (all trials)
contrast_local_list = list()
for pas in range(4):
    sub_contrast_ = analysis('local_pas_%s' % soa, 'categorize',
                             condition='local_seen',
                             query='local_undef!=True and pas==%s' % pas)
    contrast_local_list.append(sub_contrast_)
regress_local_pas = analysis('local_pas', 'regress',
                             condition=contrast_local_list)

# Local N-1 context, seen/unseen (all trials)
contrast_local_list = list()
for seen in ['seen', 'unseen']:
    truth_value = seen == 'seen'
    sub_contrast_ = analysis('local_%s' % seen, 'categorize',
                             condition='local_seen',
                             query='local_undef!=True and \
                                    seen==%s' % truth_value)
    contrast_local_list.append(sub_contrast_)
contrast_local_seen = analysis('local_seen', 'categorize',
                               condition=contrast_local_list)

# Local N-1 context, block (middle SOA trials only)
contrast_local_list = list()
blocks = ['invis', 'vis']
for b in [0, 1]:
    sub_contrast_ = analysis('local_%s' % blocks[b], 'categorize',
                             condition='local_seen',
                             query='local_undef!=True and block==%s' % b)
    contrast_local_list.append(sub_contrast_)
contrast_local_block = analysis('local_block', 'categorize',
                                condition=contrast_local_list)

# # Control contrasts

# Top vs. bottom: Cannot do because not coded in triggers

# Letter vs. digit stimulus (all present trials)
contrast_target = analysis('target', 'categorize', condition='letter_target')

# Motor response (finger to indicate letter or number)
contrast_motor = analysis('motor', 'categorize', condition='letter_resp_left')


analyses = [contrast_pst, regress_pas, regress_pas_pst, regress_pas_mid,
            contrast_seen_all, contrast_seen_pst, contrast_seen_pst_mid,
            regress_abs_seen, regress_abs_soa, contrast_block,
            regress_block_soa, contrast_block_seen,
            contrast_local, regress_local_soa, regress_local_soa_mid,
            regress_local_pas, contrast_local_seen, contrast_local_block]

# regress_block_pas

analyses = [contrast_pst, regress_pas, regress_pas_pst, regress_pas_mid,
            contrast_seen_all, contrast_seen_pst, contrast_seen_pst_mid,
            regress_abs_seen, regress_abs_soa, contrast_block, contrast_local]
analyses = [contrast_local]
# analyses = [regress_abs_soa]
