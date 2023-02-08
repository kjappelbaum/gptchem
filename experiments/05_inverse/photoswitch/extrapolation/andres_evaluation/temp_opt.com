%chk=mol.chk
%nproc=24
%mem=20GB

# PBE1PBE/6-31G(d',p') Opt Int=(Grid=SuperFineGrid)  SCF=(XQC, MaxCycle=500) SCRF=(Solvent=Ethanol)

mol1

0 1
