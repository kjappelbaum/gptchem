%Oldchk=../mol.chk
%chk=mol.chk
%nproc=24
%mem=90GB

# PBE1PBE/6-31G(d',p') Int=(Grid=SuperFineGrid)  td=(singlets, nstates=10) pop=full iop(9/40=3) GFINPUT Guess=Read Geom=Allcheck SCF=(XQC, MaxCycle=500) SCRF=(Solvent=Ethanol)

