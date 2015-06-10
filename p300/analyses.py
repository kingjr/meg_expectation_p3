from toolbox.jr_toolbox.utils import evoked_subtract, evoked_spearman

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
        dict(name=str(pas),include=dict(pas=pas))
        for pas in range(4)])

# PAS for present trials only
regress_pas_pst = dict(
    name='pas_pst', operator=evoked_spearman, conditions=[
        dict(name=str(pas), include=dict(pas=pas), exclude=dict(present=False))
        for pas in range(4)])

# PAS for present middle SOA trials only
regress_pas_mid = dict(
    name='pas_pst_mid', operator=evoked_spearman, conditions=[
        dict(name=str(idx), include=dict(pas=idx,soa=mid_soas),
            exclude=dict(present=False))
        for idx in range(4)])

# TODO can we streamline these with the new definition of seen (i.e.,
# seen/unseen whereas before it was true/false)?

# Seen vs. unseen for all trials
contrast_seen_all = dict(
    name='seen_all', operator=evoked_subtract, conditions=[
        dict(name='seen', include=dict(seen='seen')),
        dict(name='unseen', include=dict(seen='unseen'))])

# Seen vs. unseen for present trials only
contrast_seen_pst = dict(
    name='seen_pst', operator=evoked_subtract, conditions=[
        dict(name='seen', include=dict(seen='seen'), exclude=dict(present=False)),
        dict(name='unseen', include=dict(seen='unseen'), exclude=dict(present=False))])

# Seen vs. unseen for present middle SOA trials
contrast_seen_pst_mid = dict(
    name='seen_pst_mid', operator=evoked_subtract, conditions=[
        dict(name='seen', include=dict(seen='seen', soa=mid_soas)),
        dict(name='unseen', include=dict(seen='unseen', soa=mid_soas))])

# Absent, unseen present, seen present
regress_abs_seen = dict(
    name='abs_seen', operator=evoked_spearman, conditions=[
        dict(name='absent', include=dict(present=False)),
        dict(name='seen', include=dict(seen='seen'), exclude=dict(present=False)),
        dict(name='unseen', include=dict(seen='unseen'), exclude=dict(present=False))])

# 5 SOAs compared to absent
regress_soa = dict(
     name='soa', operator=evoked_spearman, conditions=[
     dict(name='absent', include=dict(present=False)),
     [dict(name=str(soa), include=dict(soa=soa)) for soa in soas]])

regress_pas_soa = list()
for soa in soas:
    regress = dict(
        name = 'pas_regress_soa' +str(soa),
        operator=evoked_subtract, conditions=[
            dict(name='pas' + str(pas) + '_soa' + str(soa),
                   include=dict(pas=pas,soa=soa), exclude=dict(present=False))
            for pas in range(4)])
    regress_pas_soa.append(regress)
# TODO add regression???

# Effects of visibility context

# Global block context (middle SOAs only)
contrast_block = dict(
    name='block',operator=evoked_subtract, condtions=[
        dict(name='vis', include=dict(block='visible',soa=mid_soas)),
        dict(name='invis', include=dict(block='invisible',soa=mid_soas))])

# Global block context, SOA (middle SOAs only)
contrast_block_soa = list()
for soa in mid_soas:
    contrast = dict(
        name='vis' +str(soa) + '-invis' + str(soa),
        operator=evoked_subtract, conditions=[
            dict(name='vis_' + str(soa), include=dict(block='visible',soa=soa)),
            dict(name='invis_' + str(soa), include=dict(block='invisible',soa=soa))])
    contrast_block_soa.append(contrast)
regress_block_soa = dict(
    name='regress_block_soa', operator=evoked_spearman,
    conditions=contrast_block_soa)

# Global block context, PAS (middle SOA, present only)
contrast_block_pas = list()
for pas in range(4):
    contrast = dict(
        name = 'vis' + str(pas) + '-invis' + str(pas),
        operator=evoked_subtract, conditions=[
            dict(name='vis_' + str(pas),
                include=dict(block='visible',pas=pas,soas=mid_soas),
                exclude=dict(present=False)),
            dict(name='invis_'+str(pas),
                include=dict(block='invisible',pas=pas,soas=mid_soas),
                exclude=dict(present=False))])
    contrast_block_pas.append(contrast)
regress_block_pas = dict(
    name='regress_block_pas', operator=evoked_spearman,
    conditions=contrast_block_pas)

# TODO Double check this!!

# Global block context, seen/unseen (middle SOAs only)
contrast_seen_block = list()
for idx, block in enumerate(['visible','invisible']):
    contrast = dict(
        name='seen' + block[:-4] + '-unseen' + block[:-4],
        operator = evoked_subtract, conditions =[
            dict(name=seen+block[:-4], include=dict(seen=seen,block=block,
                soa=mid_soas))
            for idx, seen in enumerate(['seen','unseen'])])
    contrast_seen_block.append(contrast)
contrast_seen_block = dict(
    name='contrast_seen_block', operator=evoked_subtract,
        conditions=[contrast_seen_block])

# Local N-1 context (all trials)
contrast_local = dict(
    name='local', operator=evoked_subtract, conditions=[
        dict(name='localS', include=dict(local_context='S')),
        dict(name='localU', include=dict(local_context='U'))])

# Local N-1 context, SOA (all trials)
contrast_local_soa = list()
for soa in soas:
    contrast = dict(
        name='localS'+str(soa)+'-localU'+str(soa),
        operator = evoked_subtract, conditions = [
            dict(name='localS'+str(soa), include=dict(local_context='S',soa_ttl=soa)),
            dict(name='localU'+str(soa), include=dict(local_context='U',soa_ttl=soa))])
    contrast_local_soa.append(contrast)
regress_local_soa = dict(
    name='regress_local_soa',operator=evoked_spearman,
    conditions=contrast_local_soa)

# Local N-1 context, PAS (all trials)
contrast_local_pas = list()
for pas in range(4):
    contrast = dict(
        name='localSpas'+str(pas) + '-localUpas'+str(pas),
        operator = evoked_subtract, conditions = [
            dict(name='localSpas'+str(pas), include=dict(local_context='S', pas=pas)),
            dict(name='localUpas'+str(pas), include=dict(local_context='U', pas=pas))])
    contrast_local_pas.append(contrast)
regress_local_pas = dict(
    name='regress_local_pas', operator=evoked_spearman,
    conditions=contrast_local_pas)

# Local N-1 context, seen/unseen (all trials)

contrast_local_seen = list()
for idx, local in enumerate(['S','U']):
    contrast = dict(
        name='seenlocal'+local+'-unseen'+local,
        operator = evoked_subtract, conditions = [
            dict(name=seen+local, include=dict(seen=seen, local_context=local))
            for idx, seen in enumerate(['seen','unseen'])])
    contrast_local_seen.append(contrast)
contrast_local_seen = dict(
    name='contrast_local_seen', operator=evoked_subtract,
        conditions=[contrast_local_seen])

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

analyses = [contrast_pst, regress_pas, regress_pas_pst, regress_pas_mid,
            contrast_seen_all, contrast_seen_pst, contrast_seen_pst_mid,
            regress_abs_seen, regress_soa, contrast_block, regress_block_soa,
            regress_block_pas, contrast_seen_block, contrast_local,
            regress_local_soa, regress_local_pas, contrast_local_seen,
            contrast_target, contrast_motor]

analyses = [contrast_pst, contrast_local_seen, regress_pas_pst]
