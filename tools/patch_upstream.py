#!/usr/bin/env python3
"""Patch the copy of O. Zatsarinny's DBSR sources used for the build.

Applied by CMake (cmake/FetchUpstream.cmake) to ``<build>/upstream``; the
original sources are never touched.  Every patch must match, otherwise the
build stops, so that a change upstream cannot silently disable a fix.

Patches
-------
1. Command-line / parameter-file parsing (ZCOM ``Read_*arg`` / ``Read_*par``
   and their copies inside DBSR_HF): the line buffer was ``Character(80)``
   (``Character(180)`` for some routines), so ``conf=...``, ``varied=...``,
   ``core=...`` or long file names were silently truncated, which later gave
   errors like "Bad integer for item N in list input" or "End of file".
   The buffer is raised to 2048 characters.
2. DBSR_HF: ``configuration``/``conf_AV``/``conf_LS`` (160 chars), the list of
   varied orbitals ``anit`` (80 chars) and the atom/conf strings in
   ``Def_atom`` (200 chars) are raised to 2048 characters.
3. File-name length ``ma = 80`` in the program modules -> 256
   (long paths to work directories).
5. DBSR_MCHF: ``levels=``/``weights=`` lists (Character(280)) -> 2048, and
   J-blocks without optimized levels are skipped in the diagonalization
   (DSYEVX was called with IU = 0: "parameter number 10 had an illegal value").
4. ZCOM/recup_a.f90: a comment ending with a backslash; if the file is ever
   run through cpp the next statement (``M3 = JP2(j-1)``) is lost.
6. DBSR_BREIT3 (serial and MPI): the case name (Character(80)), the scratch
   file name ``name.int_new`` (Character(40)) and the shell command
   ``cat name.int_new >> name.int_res`` (Character(80)) overflowed for case
   names longer than ~32 characters ("Fortran runtime error: End of record");
   DBSR_MCHF/get_case: the command ``dbsr_breit3 name.c`` (Character(200)).
"""
import re
import sys
from pathlib import Path

LONG = 2048

READ_ROUTINES = re.compile(r"^read_(r|i|a)(arg|par)$|^read_iarr$|^read_name$|^read_string$|^read_rval$|^read_iarray$|^read_rarray$",
                           re.I)


class PatchError(RuntimeError):
    pass


def _read(path):
    with open(path, encoding="latin-1", newline="") as f:   # keep CRLF as is
        return f.read()


def _write(path, text):
    with open(path, "w", encoding="latin-1", newline="") as f:
        f.write(text)


def split_units(text):
    """Split free-form source into (name, text) chunks at Subroutine/Function starts."""
    pat = re.compile(r"^[ \t]*(?:[\w()*]+[ \t]+)*?(?:Subroutine|Function)[ \t]+(\w+)", re.I | re.M)
    starts = [(m.start(), m.group(1)) for m in pat.finditer(text)]
    if not starts:
        return [(None, text)]
    chunks = [(None, text[:starts[0][0]])]
    for k, (pos, name) in enumerate(starts):
        end = starts[k + 1][0] if k + 1 < len(starts) else len(text)
        chunks.append((name, text[pos:end]))
    return chunks


def patch_read_routines(path):
    text = _read(path)
    out, n = [], 0
    for name, chunk in split_units(text):
        if name and READ_ROUTINES.match(name):
            new = re.sub(r"Character\((80|180)\)(\s*::\s*AS\b)", rf"Character({LONG})\2", chunk)
            new = re.sub(r"AS\((i1?):180\)", r"AS(\1:)", new)
            n += new != chunk
            chunk = new
        out.append(chunk)
    if n == 0:
        raise PatchError(f"{path}: no Read_*arg/Read_*par routine patched")
    _write(path, "".join(out))
    return n


def replace(path, old, new, count=None):
    text = _read(path)
    k = text.count(old)
    if k == 0 or (count is not None and k != count):
        raise PatchError(f"{path}: expected {count or '>=1'} occurrence(s) of {old!r}, found {k}")
    _write(path, text.replace(old, new))


def main(root):
    root = Path(root)
    lib = root / "LIBRARIES"
    prog = root / "DBSR3"

    # 1. argument / parameter buffers
    for f in [lib / "ZCOM/read_arg.f90", lib / "ZCOM/read_par.f90", prog / "DBSR_HF/dbsr_lib_zcom.f90"]:
        patch_read_routines(f)

    # trailing backslash in a comment (continuation line if ever preprocessed by cpp)
    text = _read(lib / "ZCOM/recup_a.f90")
    new = re.sub(r"(!\s+M1\s+M2)\s*\\[ \t]*(\r?\n)", r"\1\2", text)
    if new == text:
        raise PatchError("recup_a.f90: comment with trailing backslash not found")
    _write(lib / "ZCOM/recup_a.f90", new)

    # 2. DBSR_HF string lengths
    hf = prog / "DBSR_HF"
    replace(hf / "hf_MOD_dbsr_hf.f90",
            "Character(160) :: configuration = ' ', conf_AV = ' ', conf_LS = ' '",
            f"Character({LONG}) :: configuration = ' ', conf_AV = ' ', conf_LS = ' '", 1)
    replace(hf / "hf_MOD_dbsr_hf.f90", "Character(80)  :: anit = 'all'",
            f"Character({LONG}) :: anit = 'all'", 1)
    replace(hf / "hf_get_case.f90", "'orbitals to optimize:  ',anit",
            "'orbitals to optimize:  ',trim(anit)", 1)
    replace(hf / "hf_def_conf_LS.f90", "Character(160) :: conf", f"Character({LONG}) :: conf")
    replace(hf / "dbsr_lib_dbs.f90", "Character(200) :: atom, core, conf",
            f"Character({LONG}) :: atom, core, conf", 1)

    # dbsr_mchf: 'levels=' / 'weights=' lists were read into Character(280)
    t = _read(prog / "DBSR_MCHF/def_blocks.f90")
    t2 = re.sub(r"Character\(280\)", f"Character({LONG})", t)
    if t2 == t:
        raise PatchError("DBSR_MCHF/def_blocks.f90: Character(280) not found")
    _write(prog / "DBSR_MCHF/def_blocks.f90", t2)

    # dbsr_mchf: a J-block without optimized levels called DSYEVX with IU = 0
    replace(prog / "DBSR_MCHF/diag.f90",
            "      Call LAP_DSYEVX('V','L',nc,nc,HM,eval,k,info)",
            "      if(k.eq.0) Return          ! no level of this block is optimized\n"
            "      Call LAP_DSYEVX('V','L',nc,nc,HM,eval,k,info)", 1)

    # 6. dbsr_breit3 / dbsr_mchf: case names and shell commands
    for d in ("DBSR_BREIT3", "DBSR_BREIT3_MPI"):
        replace(prog / d / "mod_dbsr_breit.f90", "      Character(80) :: name",
                "      Character(256) :: name", 1)
        replace(prog / d / "mod_dbsr_breit.f90", "Character(40) :: AF_i = 'int_new'",
                "Character(256) :: AF_i = 'int_new'", 1)
    replace(prog / "DBSR_BREIT3/dbsr_breit.f90", "Character(80) :: cline", f"Character({LONG}) :: cline", 1)
    replace(prog / "DBSR_BREIT3_MPI/dbsr_breit_mpi.f90", "Character(80) :: cline",
            f"Character({LONG}) :: cline", 1)
    replace(prog / "DBSR_MCHF/get_case.f90", "Character(200) :: A_core, A_conf, AC",
            f"Character({LONG}) :: A_core, A_conf, AC", 1)

    # 3. file-name length in program modules
    n = 0
    for f in sorted(prog.rglob("*.f90")):
        t = _read(f)
        t2 = re.sub(r"(Integer,\s*parameter\s*::\s*ma\s*=\s*)80\b", r"\g<1>256", t, flags=re.I)
        if t2 != t:
            _write(f, t2)
            n += 1
    if n == 0:
        raise PatchError("no 'ma = 80' parameter found in DBSR3")
    print(f"patch_upstream: patched {root}")


if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except PatchError as e:
        sys.exit(f"patch_upstream: {e}")
