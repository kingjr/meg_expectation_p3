from toolbox.jr_toolbox.utils import evoked_subtract, evoked_spearman

# TODO check these contrasts
# before name means syntax error
# Soa only defined for target-present trials.
# Need to use soa_ttl to include absent.

soas = [17, 33, 50, 67, 83]
mid_soas = [33, 50, 67]

# Key contrasts for replication

# Present vs. absent
contrast_pst = dict(
    name='presence', operator=evoked_subtract, conditions=[
        dict(name='present', include=dict(present=True)),
        dict(name='absent', include=dict(present=False))])

# PAS for all trials
regress_pas = dict(
    name='pas_all', operator=evoked_spearman, conditions=[
        dict(name='0', include=dict(pas=0)),
        dict(name='1', include=dict(pas=1)),
        dict(name='2', include=dict(pas=2)),
        dict(name='3', include=dict(pas=3))])

# PAS for present trials only
regress_pas_pst = dict(
    name='pas_pst', operator=evoked_spearman, conditions=[
        dict(name=str(idx), include=dict(pas=idx), exclude=dict(present=False))
        for idx in range(4)])

# PAS for present middle SOA trials only
regress_pas_mid = dict(
    name='pas_pst_mid', operator=evoked_spearman, conditions=[
        dict(name=str(idx), include=dict(pas=idx,soa=[33,50,67]),
            exclude=dict(present=False))
        for idx in range(4)])

# Seen vs. unseen for all trials
contrast_seen_all = dict(
    name='seen_all', operator=evoked_subtract, conditions=[
        dict(name='seen', include=dict(seen=True)),
        dict(name='unseen', include=dict(seen=False))])

# Seen vs. unseen for present trials only
contrast_seen_pst = dict(
    name='seen_pst', operator=evoked_subtract, conditions=[
        dict(name='seen', include=dict(seen=True), exclude=dict(present=False)),
        dict(name='unseen', include=dict(seen=False), exclude=dict(present=False))])

# Seen vs. unseen for present middle SOA trials
contrast_seen_pst_mid = dict(
    name='seen_pst_mid', operator=evoked_subtract, conditions=[
        dict(name='seen', include=dict(seen=True, soa=[33,50,67])),
        dict(name='unseen', include=dict(seen=False, soa=[33,50,67]))])

# Absent, unseen present, seen present
regress_abs_seen = dict(
    name='abs_seen', operator=evoked_spearman, conditions=[
        dict(name='absent', include=dict(present=False)),
        dict(name='seen', include=dict(seen=True), exclude=dict(present=False)),
        dict(name='unseen', include=dict(seen=False), exclude=dict(present=False))])

# TODO help??
# 5 SOAs compared to absent

regress_soa = dict(
     name='soa', operator=evoked_spearman, conditions=[
     dict(name='absent', include=dict(present=False)),
     [dict(name=str(soa), include=dict(soa=soa)) for soa in soas]])

# TODO what is idx in this case?
regress_pas_soa = list()
for idx, soa in enumerate([17, 33, 50, 67, 83]):
    regress = dict(
        name = 'pas_regress_soa' +str(soa),
        operator=evoked_subtract, conditions=[
            dict(name='pas' + str(pas) + '_soa' + str(soa),
                   include=dict(pas=pas,soa=soa), exclude=dict(present=False))
            for pas in range(4)])
    regress_pas_soa.append(regress)

## Effects of visibility context

# Global block context (middle SOAs only)
contrast_block = dict(
    name='block',operator=evoked_subtract, condtions=[
        dict(name='vis', include=dict(block='visible',soa=[33,50,67])),
        dict(name='invis', include=dict(block='invisible',soa=[33,50,67]))])

## Loops starting here

# Global block context, SOA (middle SOAs only)
contrast_block_soa = list()
for soa in [33, 50, 67]:
    contrast = dict(
        name='vis' +str(soa) + '-invis' + str(soa),
        operator=evoked_subtract, conditions=[
            dict(name='vis_' + str(soa), include=dict(block='visible',soa=soa)),
            dict(name='invis_' + str(soa), include=dict(block='invisible',soa=soa))])
    contrast_block_soa.append(contrast)
regression_block_soa = dict(
    name='regress_block_soa', operator=evoked_spearman,
    conditions=contrast_block_soa)

# # Global block context, PAS (middle SOA, presents only)
# contrast_block_pas = list()
# for idx, pas in range(4)
#
#

contrast_block_pas0=dict(
    name = 'vispas0-invispas0', operator=evoked_subtract, conditions=[
        dict(name='vis_pas0',
        include=dict(block='visible', pas=0, soa=[33,50,67]),
        exclude=dict(present=False)),
        dict(name='invis_pas0',
        include=dict(block='invisible', pas=0, soa=[33,50,67]),
        exclude=dict(present=False))])
contrast_block_pas1=dict(
    name = 'vispas1-invispas1', operator=evoked_subtract, conditions=[
        dict(name='vis_pas1',
        include=dict(block='visible', pas=1, soa=[33,50,67]),
        exclude=dict(present=False)),
        dict(name='invis_pas1',
        include=dict(block='invisible', pas=1, soa=[33,50,67]),
        exclude=dict(present=False))])
contrast_block_pas2=dict(
    name = 'vispas2-invispas2', operator=evoked_subtract, conditions=[
        dict(name='vis_pas2',
        include=dict(block='visible', pas=2, soa=[33,50,67]),
        exclude=dict(present=False)),
        dict(name='invis_pas2',
        include=dict(block='invisible', pas=2, soa=[33,50,67]),
        exclude=dict(present=False))])
contrast_block_pas3=dict(
    name = 'vispas3-invispas3', operator=evoked_subtract, conditions=[
        dict(name='vis_pas3',
        include=dict(block='visible', pas=3, soa=[33,50,67]),
        exclude=dict(present=False)),
        dict(name='invis_pas3',
        include=dict(block='invisible', pas=3, soa=[33,50,67]),
        exclude=dict(present=False))])
regress_blockXsoa = dict(
    name = 'block_X_soa', operator=evoked_spearman,conditions=[
        contrast_block_pas0, contrast_block_pas1, contrast_block_pas2,
        contrast_block_pas3])

# Global block context, seen/unseen (middle SOAs only)
contrast_seenvis = dict(
    name='seenvis-unseenvis', operator=evoked_subtract, condtions=[
        dict(name='seen_vis',
        include=dict(seen=True, block='visible', soa=[33,50,67]),
        exclude=dict(present=False)),
        dict(name='unseen_vis',
        include=dict(seen=False, block='visible', soa=[33,50,67]),
        exclude=dict(present=False))])
contrast_seeninvis = dict(
    name='seeninvis-unseeninvis', operator=evoked_subtract, condtions=[
        dict(name='seen_invis',
        include=dict(seen=True, block='invisible', soa=[33,50,67]),
        exclude=dict(present=False)),
        dict(name='unseen_invis',
        include=dict(seen=False, block='invisible', soa=[33,50,67]),
        exclude=dict(present=False))])
contrast_seenXblock = dict(
    name='seen_X_block', operator=evoked_subtract, conditions=[
        contrast_seenvis, contrast_seeninvis])

# Local N-1 context (all trials)
contrast_local = dict(
    name='local', operator=evoked_subtract, conditions=[
        dict(name='localS', include=dict(local_context='S')),
        dict(name='localU', include=dict(local_context='U'))])

# Local N-1 context, SOA (all trials)
contrast_local17 = dict(
    name='localS17-localU17', operator=evoked_subtract,conditions=[
        dict(name='localS17',include=dict(local_context='S',soa_ttl=17)),
        dict(name='localU17',include=dict(local_context='U',soa_ttl=17))])
contrast_local33 = dict(
    name='localS33-localU33', operator=evoked_subtract,conditions=[
        dict(name='localS33',include=dict(local_context='S',soa_ttl=33)),
        dict(name='localU33',include=dict(local_context='U',soa_ttl=33))])
contrast_local50 = dict(
    name='localS50-localU50', operator=evoked_subtract,conditions=[
        dict(name='localS50',include=dict(local_context='S',soa_ttl=50)),
        dict(name='localU50',include=dict(local_context='U',soa_ttl=50))])
contrast_local67 = dict(
    name='localS67-localU67', operator=evoked_subtract,conditions=[
        dict(name='localS67',include=dict(local_context='S',soa_ttl=67)),
        dict(name='localU67',include=dict(local_context='U',soa_ttl=67))])
contrast_local83 = dict(
    name='localS83-localU83', operator=evoked_subtract,conditions=[
        dict(name='localS83',include=dict(local_context='S',soa_ttl=83)),
        dict(name='localU83',include=dict(local_context='U',soa_ttl=83))])
regress_localXsoa = dict(
    name='local_X_soa', operator=evoked_spearman, conditions=[
        contrast_local17, contrast_local33, contrast_local50, contrast_local67,
        contrast_local83])


# Local N-1 context, PAS (all trials)
contrast_localpas0 = dict(
    name='localSpas0-localUpas0', operator=evoked_subtract,conditions=[
        dict(name='localSpas0',include=dict(local_context='S',pas=0)),
        dict(name='localUpas0',include=dict(local_context='U',pas=0))])
contrast_localpas1 = dict(
    name='localSpas1-localUpas1', operator=evoked_subtract,conditions=[
        dict(name='localSpas1',include=dict(local_context='S',pas=1)),
        dict(name='localUpas1',include=dict(local_context='U',pas=1))])
contrast_localpas2 = dict(
    name='localSpas2-localUpas2', operator=evoked_subtract,conditions=[
        dict(name='localSpas2',include=dict(local_context='S',pas=2)),
        dict(name='localUpas2',include=dict(local_context='U',pas=2))])
contrast_localpas3 = dict(
    name='localSpas3-localUpas3', operator=evoked_subtract,conditions=[
        dict(name='localSpas3',include=dict(local_context='S',pas=3)),
        dict(name='localUpas3',include=dict(local_context='U',pas=3))])
regress_localXpas = dict(
    name='local_X_pas', operator=evoked_spearman, conditions=[
    contrast_localpas0, contrast_localpas1, contrast_localpas2,
    contrast_localpas3])

# Local N-1 context, seen/unseen (all trials) # TODO flip this order to match others??
contrast_seenlocalS = dict(
    name='seenS-unseenS', operator=evoked_subtract, conditions=[
        dict(name='seen_S', include=dict(seen=True, local_context='S')),
        dict(name='unseen_S', include=dict(seen=False, local_context='S'))])
contrast_seenlocalU = dict(
    name='seenU-unseenU', operator=evoked_subtract, conditions=[
        dict(name='seen_U', include=dict(seen=True, local_context='U')),
        dict(name='unseen_U', include=dict(seen=False, local_context='U'))])
contrast_seenXlocal = dict(
    name='seen_X_local', operator=evoked_subtract, conditions=[
        contrast_seenlocalS, contrast_seenlocalU])

# TODO Local N-1 context, local N-2 context (all trials)
# ??? Or regression with all four combinations?

## Control contrasts

# Top vs. bottom: Cannot do because not coded in triggers

# Letter vs. digit stimulus (all present trials)
contrast_target = dict(
    name = 'target', operator=evoked_subtract, conditions=[
        dict(name='letter', include=dict(target='letter')),
        dict(name='number', include=dict(target='number'))])

# Motor response (finger to indicate letter or number)
contrast_motor = dict(
    name='motor_resp', operator=evoked_subtract, conditions=[
        dict(name='left', include=dict(letter_resp='left')),
        dict(name='right', include=dict(letter_resp='right'))])


analyses = [contrast_pst, contrast_seenXlocal, regress_pas_pst]
