"""Parse Zatsarinny makefiles -> cmake/DBSRSources.cmake (source lists)."""
import re, os, sys
ROOT = sys.argv[1]
def mvars(path):
    txt = open(path).read().replace('\\\n', ' ')
    out = {}
    for line in txt.splitlines():
        m = re.match(r'^\s*([A-Z]+)\s*=(.*)$', line)
        if m: out[m.group(1)] = m.group(2).split('#')[0].split()
    return out
def mk(d):
    for n in ('Makefile', 'makefile'):
        if os.path.exists(os.path.join(ROOT, d, n)): return os.path.join(ROOT, d, n)
L = []
def emit(name, files):
    L.append(f"set({name}\n    " + "\n    ".join(files) + ")\n")
for lib in ['ZCOM', 'ZCONFJJ', 'ZCONFLS', 'DBS']:
    v = mvars(mk('LIBRARIES/' + lib))
    emit('SRC_' + lib, v.get('C', []) + v['S'] + v.get('SS', []))
progs = {'DBSR_HF': 'S', 'DBSR_PREP3': 'S', 'DBSR_BREIT3': 'S', 'DBSR_MAT3': 'S',
         'DBSR_HD3': 'S', 'DBSR_MULT3': 'S', 'DBSR_DMAT3': 'S', 'DBSR_POL3': 'S',
         'DBSR_CI3': 'S', 'DBSR_MCHF': 'S'}
for p, key in progs.items():
    emit('SRC_' + p, mvars(mk('DBSR3/' + p))[key])
v = mvars(mk('DBSR3/DBSR_CONF3'))
emit('SRC_DBSR_CONF3', v['S'] + v['SN']); emit('SRC_DBSR_CONF3_MPI', v['S'] + v['SM'])
emit('SRC_DBSR_BREIT3_MPI', mvars(mk('DBSR3/DBSR_BREIT3_MPI'))['S'])
v = mvars(os.path.join(ROOT, 'DBSR3/DBSR_MAT3/Makefile_mpi'))
emit('SRC_DBSR_MAT3_MPI', v['SO'] + v['SM'] + v['SB'])
emit('SRC_DBSR_HD3_MPI', mvars(mk('DBSR3/DBSR_HD3_MPI'))['S'])
open(sys.argv[2], 'w').write("# Generated from upstream makefiles by tools/gen_sources.py\n\n" + "\n".join(L))
