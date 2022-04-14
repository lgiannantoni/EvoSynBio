import re
import sys

import PyBoolNet.StateTransitionGraphs as STGs
import PyBoolNet.FileExchange as FE
import PyBoolNet.ModelChecking as MC

def right_vars(t):
    l = set()
    for sublist in t:
        for d in sublist:
            #print(d)
            l |= set(d.keys())
    return l
    #return [item for sublist in t for item in sublist]

def visit(left, right_expr, keep, visited):
    right_expr -= keep
    #if right_expr == {left}:
    if left in right_expr:
        keep |= {left}
        right_expr -= {left}
        visited |= {left}
    for left in right_expr:
        if left not in visited:
            visited |= {left}
            right = right_vars(primes[left])
            keep |= visit(left, right, keep, visited)
    return keep

if __name__ == "__main__":
    f = "regan2020.bnet"
    with open(f, "r") as fin:
        print(fin.name)

        right = list()
        left = list()

        for _line in fin:
            if not _line[0] == '#':
                _line = _line.replace('\n', '').replace('\t', '').replace(' ', '')
                try:
                    _l, _r = _line.split(',')
                    left.append(_l)
                    right.extend(filter(None, re.split("[|&()!]+", _r)))
                except Exception as e:
                    left.append(_line)
        left = set(left)
        right = set(right)
        big_no_no = [_r for _r in right if _r not in left]
        assert not big_no_no, f"The following {len(big_no_no)} nodes are undefined: {big_no_no}"

    with open(f, "r") as fin:
        print(fin.name)
        bnet = fin.read()
        #print(bnet, type(bnet))
        assert bnet, "bnet is None"

        primes = FE.bnet2primes(bnet)
        #stg = STGs.primes2stg(primes, "synchronous")
        #STGs.stg2image(stg, "example18_stg.pdf")
        #print(primes)

        #_left = 'GranularKeratinocyte' # 'PluripotentStemCell' # 'GF' # 'Stress_Fibers' #
        #_right = right_vars(primes[_left])
        #res = visit(_left, _right, keep=set(), visited=set())
        #print(f"res ({len(res)} nodes): {res}")

    print(f"Model {fin.name} should be ok")



# test con target cell_state_PluripotentStemCell
# res (45 nodes): {'Oct4', 'pAPC', 'Trail', 'Replication', 'TGFbeta', 'EGF', 'Calcitriol', 'Cdc6', 'CellDensity_Low',
#                  'extracellCa', 'CyclinB', 'CyclinA', 'f4N_DNA', 'FGF2', 'OCT4', 'SOX2', 'cJun', 'SMAD4', 'Y27632',
#                  'STAT3', 'Delta', 'FGF7', 'CellDensity_High', 'AKT', 'Stiff_ECM', 'Cdk1', 'UbcH10', 'GF_High',
#                  'Inhibitin_beta_B', 'A_Kinetochores', 'CyclinD1', 'E2F1', 'Insulin', 'p110_H', 'Casp3', 'Activin',
#                  'GF', 'SMAD158', 'ECM', 'U_Kinetochores', 'BMP4', 'DAPT', 'WNT', 'Plk1_H', 'T3'}
#
# inputs (in model_validation.odt): Inhibitin_beta_B, Activin, FGF2, Integrins; (n.b. è Integrin)
#
# 1. Integrin non è tra gli input
#       * non è un input: Integrin <- (ECM & !Notch) | (T3 & !Notch) | (p63 & !Notch)
#
# =================================================================================================================================================
#
# test con target cell_state_suspended_PluripotentStemCell
# res (1 nodes): {'cell_state_suspended_PluripotentStemCell'}
#
# inputs (in model_validation.odt): BMP4, Y27632
#
# 1. Risultato giustificato dal fatto che la riga con cell_state_suspended_PluripotentStemCell <- Sox2 & Oct4 & Nanog & !cell_process_Apoptosis
# è stata commentata, ma cell_state_suspended_PluripotentStemCell viene ancora usato nel calcolo di cell_state_SurfaceEpithelialCommitment
#
# test con quella riga scommentata:
# res (47 nodes): {'Replication', 'RetinoicAcid', 'SMAD4', 'GF', 'CyclinD1', 'CyclinB', 'U_Kinetochores', 'f4N_DNA',
#                  'FGF7', 'A_Kinetochores', 'UbcH10', 'Cdk1', 'Trail', 'WNT', 'Oct4', 'AKT', 'GF_High', 'SMAD158',
#                  'p110_H', 'Cdc6', 'ECM', 'E2F1', 'CellDensity_High', 'SOX2', 'extracellCa', 'Plk1_H', 'OCT4', 'T3',
#                  'Calcitriol', 'TGFbeta', 'FGF2', 'Insulin', 'Inhibitin_beta_B', 'Stiff_ECM', 'Delta', 'Activin',
#                  'STAT3', 'Y27632', 'CellAdhesion', 'CellDensity_Low', 'pAPC', 'Casp3', 'EGF', 'CyclinA', 'cJun',
#                  'DAPT', 'BMP4'}
#
# 1. OK: tutti i nodi indicati come input sono effettivamente input della rete
# 2. KO: il nodo target dipende da ulteriori 45 nodi
#
# I test successivi sono tutti sul bnet con la riga relativa a cell_state_suspended_PluripotentStemCell commentata
# n.b. da model_validation.odt mancano gli input per passare da cell_state_suspended_PluripotentStemCell a cell_state_SurfaceEpithelialCommitment
# ma nel bnet è stata commentata l'equazione relativa a cell_state_suspended_PluripotentStemCell:
# quale dei due stati non dobbiamo utilizzare?
#
# =================================================================================================================================================
#
# test con target cell_state_SurfaceEpithelialCommitment
# res (2 nodes): {'RetinoicAcid', 'cell_state_suspended_PluripotentStemCell'}
#
# inputs (in model_validation.odt): non presenti
#
# 1. cell_state_suspended_PluripotentStemCell non dovrebbe essere nell'equazione, perché abbiamo deciso di realizzare
# la macchina a stati in un simulatore a parte (i.e. fuori dal modello booleano).
#
# test con target cell_state_KeratinocyteLineageSelection
# res (47 nodes): {'Activin', 'AKT', 'Insulin', 'FGF7', 'BMP4', 'ECM', 'Stiff_ECM', 'DAPT', 'A_Kinetochores', 'SMAD158',
#                  'OCT4', 'Delta', 'RetinoicAcid', 'Cdc6', 'extracellCa', 'CellDensity_Low', 'Trail', 'T3', 'Replication',
#                  'E2F1', 'Casp3', 'Cdk1', 'Calcitriol', 'FGF2', 'STAT3', 'f4N_DNA', 'TGFbeta', 'CellAdhesion',
#                  'U_Kinetochores', 'Plk1_H', 'CyclinD1', 'GF', 'WNT', 'cJun', 'CyclinA', 'CellDensity_High', 'EGF',
#                  'Oct4', 'pAPC', 'UbcH10', 'Y27632', 'CyclinB', 'Inhibitin_beta_B', 'GF_High', 'SMAD4', 'SOX2', 'p110_H'}
#
# inputs (in model_validation.odt): Insulin, RetinoicAcid, Adenine, CholeraToxin, T3
#
# 1. Adenine non è tra gli input risultanti
#       * entra solo nel calcolo di cell_process_SelfRenewal <- Adenine | Oct4 | Sox2 | Nanog
#       * cell_process_SelfRenewal a sua volta non è input di altre equazioni, quindi è "scollegato" dal resto della rete
# 2. CholeraToxin non è tra gli input risultanti
#       * entra solo nel calcolo di AC <- CholeraToxin
#       * AC entra solo nel calcolo di cAMP <- AC
#       * cAMP non è input di altre equazioni
#
# =================================================================================================================================================
#
# test con target cell_state_BasalKeratinocyte
# res (47 nodes): {'Cdk1', 'CyclinA', 'U_Kinetochores', 'SMAD4', 'UbcH10', 'FGF7', 'RetinoicAcid', 'STAT3', 'OCT4',
#                  'cJun', 'Plk1_H', 'Cdc6', 'Stiff_ECM', 'WNT', 'Replication', 'CellAdhesion', 'Oct4', 'FGF2', 'BMP4',
#                  'AKT', 'SOX2', 'Calcitriol', 'Trail', 'Activin', 'p110_H', 'Insulin', 'Casp3', 'GF', 'ECM',
#                  'Inhibitin_beta_B', 'Y27632', 'DAPT', 'A_Kinetochores', 'CellDensity_High', 'E2F1', 'TGFbeta',
#                  'SMAD158', 'extracellCa', 'EGF', 'CellDensity_Low', 'CyclinB', 'GF_High', 'Delta', 'pAPC', 'T3',
#                  'CyclinD1', 'f4N_DNA'}
#
# inputs (in model_validation.odt): Integrin, Insulin, RetinoicAcid, Adenine, CholeraToxin, T3, BMP4, EGF
#
# 1. Integrin non è tra gli input risultanti (v. sopra)
# 2. Adenine non è tra gli input risultanti (v. sopra)
# 3. CholeraToxin non è tra gli input risultanti (v. sopra)
#
# =================================================================================================================================================
#
# test con target cell_state_SpinousKeratinocyte
# res (48 nodes): {'ECM', 'CellDensity_High', 'CellAdhesion', 'RetinoicAcid', 'FGF7', 'SMAD4', 'p110_H', 'SOX2',
#                  'Inhibitin_beta_B', 'U_Kinetochores', 'EGF', 'pAPC', 'Insulin', 'BMP4', 'A_Kinetochores', 'UbcH10',
#                  'TGFbeta', 'WNT', 'Oct4', 'T3', 'E2F1', 'CellDensity_Low', 'Stiff_ECM', 'Cdc6', 'OCT4', 'DAPT',
#                  'Y27632', 'GF_High', 'Activin', 'Plk1_H', 'FRA1', 'CyclinB', 'AKT', 'FGF2', 'Cdk1', 'STAT3', 'Trail',
#                  'CyclinA', 'Casp3', 'f4N_DNA', 'cJun', 'Delta', 'CyclinD1', 'SMAD158', 'extracellCa', 'Calcitriol',
#                  'Replication', 'GF'}
#
# inputs (in model_validation.odt): extracellCa, Delta, Calcitriol
#
# 1. OK: tutti i nodi indicati come input sono effettivamente input della rete
# 2. KO: il nodo target dipende da ulteriori 45 nodi
#
# =================================================================================================================================================
#
# test con target cell_state_GranularKeratinocyte
# res (49 nodes): {'FGF2', 'Delta', 'U_Kinetochores', 'OCT4', 'CellAdhesion', 'Y27632', 'pAPC', 'CyclinD1', 'Plk1_H',
#                  'DAPT', 'CellDensity_High', 'GF_High', 'CyclinB', 'T3', 'FGF7', 'p110_H', 'extracellCa', 'Casp3',
#                  'Replication', 'Activin', 'UbcH10', 'E2F1', 'Stiff_ECM', 'Cdc6', 'STAT3', 'GF', 'BMP4', 'WNT', 'RPBJ',
#                  'RetinoicAcid', 'Trail', 'A_Kinetochores', 'SMAD158', 'Calcitriol', 'CellDensity_Low', 'SOX2', 'Oct4',
#                  'f4N_DNA', 'Insulin', 'TGFbeta', 'EGF', 'SMAD4', 'AKT', 'Inhibitin_beta_B', 'CyclinA', 'cJun', 'ECM',
#                  'Cdk1', 'FRA1'}
#
# inputs (in model_validation.odt): extracellCa, Delta, Calcitriol
#
# 1. OK: tutti i nodi indicati come input sono effettivamente input della rete
# 2. KO: il nodo target dipende da ulteriori 46 nodi




