cmake_minimum_required(VERSION 3.18)
project(pydbsr_fortran LANGUAGES Fortran C)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(DBSR_ENABLE_MPI        "Build MPI versions (dbsr_conf3_mpi, dbsr_breit3_mpi, dbsr_mat3_mpi)" OFF)
option(DBSR_ENABLE_SCALAPACK  "Build dbsr_hd3_mpi (needs ScaLAPACK, implies MPI)" OFF)
option(DBSR_BUILD_UTILS       "Build Zatsarinny utilities (jcfile, sum_hh_jj, dbound_*, ...)" ON)
option(DBSR_LARGE_MEMORY      "Compile with -mcmodel=medium (static data > 2 GB)" ON)
option(DBSR_DEFAULT_REAL8     "Promote default REAL to 8 bytes (-fdefault-real-8)" OFF)
option(DBSR_NATIVE            "Optimise for the build machine (-march=native)" OFF)
option(DBSR_DEBUG_CHECKS      "Bounds checks and backtraces" OFF)
option(DBSR_STATIC_RUNTIME    "Link libgfortran/libgcc and BLAS/LAPACK statically (portable wheels)" OFF)
set(DBSR_UPSTREAM_DIR "" CACHE PATH
    "Directory with upstream DBSR3/, LIBRARIES/, UTILS/ (otherwise they are downloaded)")
set(DBSR_INSTALL_BINDIR "pydbsr/bin" CACHE STRING "Where executables are installed")
set(DBSR_INSTALL_LIBDIR "pydbsr/lib" CACHE STRING "Where the Coulomb-function library is installed")

# The sources are plain Fortran: never run them through cpp (the Ninja
# generator does so by default for scanning; a trailing backslash in a comment
# of ZCOM/recup_a.f90 then swallows the next statement -> ZRECUP errors).
set(CMAKE_Fortran_PREPROCESS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---------------------------------------------------------------------------
# Upstream sources (O. Zatsarinny, github.com/zatsaroi) -> patched copy
# ---------------------------------------------------------------------------
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FetchUpstream.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/DBSRSources.cmake)

set(U ${DBSR_SRC})               # patched upstream tree
set(LIBS_DIR ${U}/LIBRARIES)
set(PROG_DIR ${U}/DBSR3)
set(UTIL_DIR ${U}/UTILS)

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------
set(DBSR_FFLAGS "")
if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
  list(APPEND DBSR_FFLAGS -ffree-line-length-none -fallow-argument-mismatch
                          -fno-range-check -Wno-all)
  if(DBSR_DEFAULT_REAL8)
    list(APPEND DBSR_FFLAGS -fdefault-real-8 -fdefault-double-8)
  endif()
  if(DBSR_DEBUG_CHECKS)
    list(APPEND DBSR_FFLAGS -fcheck=bounds -fbacktrace -g)
  endif()
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
  if(DBSR_DEFAULT_REAL8)
    list(APPEND DBSR_FFLAGS -real-size 64)
  endif()
  if(DBSR_DEBUG_CHECKS)
    list(APPEND DBSR_FFLAGS -check bounds -traceback -g)
  endif()
endif()
if(DBSR_LARGE_MEMORY AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64" AND NOT APPLE)
  list(APPEND DBSR_FFLAGS -mcmodel=medium)
endif()
if(DBSR_NATIVE)
  list(APPEND DBSR_FFLAGS -march=native)
endif()

if(DBSR_STATIC_RUNTIME)
  set(BLA_STATIC ON)
  if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    add_link_options(-static-libgfortran -static-libgcc)
    if(NOT APPLE)
      add_link_options(-static-libquadmath)
    endif()
  endif()
endif()
find_package(LAPACK REQUIRED)

# ---------------------------------------------------------------------------
# Libraries.  Upstream compiles all libraries into one module directory,
# so do we (modules of different libraries do not clash).
# ---------------------------------------------------------------------------
set(LIB_MODDIR ${CMAKE_BINARY_DIR}/mod_libs)

function(dbsr_library name dir)
  list(TRANSFORM ARGN PREPEND ${dir}/ OUTPUT_VARIABLE srcs)
  add_library(${name} STATIC ${srcs})
  set_target_properties(${name} PROPERTIES Fortran_MODULE_DIRECTORY ${LIB_MODDIR}
                                           POSITION_INDEPENDENT_CODE ON)
  target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:${DBSR_FFLAGS}>)
  target_include_directories(${name} PUBLIC ${LIB_MODDIR})
endfunction()

dbsr_library(zcom    ${LIBS_DIR}/ZCOM    ${SRC_ZCOM})
dbsr_library(zconfjj ${LIBS_DIR}/ZCONFJJ ${SRC_ZCONFJJ})
dbsr_library(zconfls ${LIBS_DIR}/ZCONFLS ${SRC_ZCONFLS})
dbsr_library(dbs     ${LIBS_DIR}/DBS     ${SRC_DBS})
target_link_libraries(zconfjj PUBLIC zcom)
target_link_libraries(zconfls PUBLIC zcom)
target_link_libraries(dbs     PUBLIC zconfjj zcom)
# upstream link order: $(DBS) $(ZCONFJJ) $(ZCOM) $(LAPACK)

set(DBSR_PROGRAMS "")

# program-local modules must shadow the library ones -> own module dir first
function(dbsr_program name dir)
  cmake_parse_arguments(P "MPI" "" "SOURCES;LIBS" ${ARGN})
  list(TRANSFORM P_SOURCES PREPEND ${dir}/ OUTPUT_VARIABLE srcs)
  add_executable(${name} ${srcs})
  set(moddir ${CMAKE_BINARY_DIR}/mod_${name})
  set_target_properties(${name} PROPERTIES Fortran_MODULE_DIRECTORY ${moddir}
                                           LINKER_LANGUAGE Fortran)
  target_include_directories(${name} BEFORE PRIVATE ${moddir})
  target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:${DBSR_FFLAGS}>)
  target_link_libraries(${name} PRIVATE ${P_LIBS} ${LAPACK_LIBRARIES})
  if(P_MPI)
    target_link_libraries(${name} PRIVATE MPI::MPI_Fortran)
  endif()
  set(DBSR_PROGRAMS ${DBSR_PROGRAMS} ${name} PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------
# DBSR3 programs
# ---------------------------------------------------------------------------
dbsr_program(dbsr_hf     ${PROG_DIR}/DBSR_HF     SOURCES ${SRC_DBSR_HF})
dbsr_program(dbsr_prep3  ${PROG_DIR}/DBSR_PREP3  SOURCES ${SRC_DBSR_PREP3}  LIBS dbs zconfjj zcom)
dbsr_program(dbsr_conf3  ${PROG_DIR}/DBSR_CONF3  SOURCES ${SRC_DBSR_CONF3}  LIBS zconfjj zcom)
dbsr_program(dbsr_breit3 ${PROG_DIR}/DBSR_BREIT3 SOURCES ${SRC_DBSR_BREIT3} LIBS zconfjj zcom)
dbsr_program(dbsr_mat3   ${PROG_DIR}/DBSR_MAT3   SOURCES ${SRC_DBSR_MAT3}   LIBS dbs zconfjj zcom)
dbsr_program(dbsr_hd3    ${PROG_DIR}/DBSR_HD3    SOURCES ${SRC_DBSR_HD3}    LIBS dbs zconfjj zcom)
# DSYEVD with 64-bit integers from a run-time loaded ILP64 LAPACK (large matrices);
# pure Fortran: mixing C objects made the linker pick up conda's C sysroot with a system gfortran
target_sources(dbsr_hd3 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/fortran/pydbsr_ilp64.f90)
target_link_libraries(dbsr_hd3 PRIVATE ${CMAKE_DL_LIBS})
dbsr_program(dbsr_mult3  ${PROG_DIR}/DBSR_MULT3  SOURCES ${SRC_DBSR_MULT3}  LIBS zconfjj zcom)
dbsr_program(dbsr_dmat3  ${PROG_DIR}/DBSR_DMAT3  SOURCES ${SRC_DBSR_DMAT3}  LIBS dbs zconfjj zcom)
dbsr_program(dbsr_pol3   ${PROG_DIR}/DBSR_POL3   SOURCES ${SRC_DBSR_POL3}   LIBS dbs zconfjj zcom)
dbsr_program(dbsr_ci3    ${PROG_DIR}/DBSR_CI3    SOURCES ${SRC_DBSR_CI3}    LIBS dbs zconfjj zcom)
dbsr_program(dbsr_mchf   ${PROG_DIR}/DBSR_MCHF   SOURCES ${SRC_DBSR_MCHF}   LIBS dbs zconfjj zcom)

if(DBSR_ENABLE_MPI OR DBSR_ENABLE_SCALAPACK)
  find_package(MPI REQUIRED COMPONENTS Fortran)
  dbsr_program(dbsr_conf3_mpi  ${PROG_DIR}/DBSR_CONF3      MPI SOURCES ${SRC_DBSR_CONF3_MPI}  LIBS zconfjj zcom)
  dbsr_program(dbsr_breit3_mpi ${PROG_DIR}/DBSR_BREIT3_MPI MPI SOURCES ${SRC_DBSR_BREIT3_MPI} LIBS zconfjj zcom)
  dbsr_program(dbsr_mat3_mpi   ${PROG_DIR}/DBSR_MAT3       MPI SOURCES ${SRC_DBSR_MAT3_MPI}   LIBS dbs zconfjj zcom)
endif()
if(DBSR_ENABLE_SCALAPACK)
  find_library(SCALAPACK_LIB NAMES scalapack scalapack-openmpi scalapack-mpich REQUIRED)
  dbsr_program(dbsr_hd3_mpi ${PROG_DIR}/DBSR_HD3_MPI MPI SOURCES ${SRC_DBSR_HD3_MPI}
               LIBS dbs zconfjj zcom ${SCALAPACK_LIB})
endif()

# ---------------------------------------------------------------------------
# Utilities (one source file each)
# ---------------------------------------------------------------------------
function(dbsr_util name dir)
  dbsr_program(${name} ${dir} SOURCES ${name}.f90 ${ARGN})
  set(DBSR_PROGRAMS ${DBSR_PROGRAMS} PARENT_SCOPE)
endfunction()

if(DBSR_BUILD_UTILS)
  foreach(u cm_j coef_jj genjconf genjterm jcfile jfile jj_list mult_jj order_jj)
    dbsr_util(${u} ${UTIL_DIR}/CONF_JJ LIBS zconfjj zcom)
  endforeach()
  foreach(u bsw_rw dbound_bsw dbsr_merge dbsw123 rw_bsw)
    dbsr_util(${u} ${UTIL_DIR}/DBSR_UTILS LIBS dbs zconfjj zcom)
  endforeach()
  foreach(u rw123 rw_dat zf_cc)
    dbsr_util(${u} ${UTIL_DIR}/DBSR_UTILS LIBS zconfjj zcom)
  endforeach()
  dbsr_util(dbound_tab ${UTIL_DIR}/DBSR_UTILS LIBS zcom)
  foreach(u f_values_jj get1tdonejj gettdonejj sec_dif_JJ sec_dif_JJ_ampl targ_jj_exp)
    dbsr_util(${u} ${UTIL_DIR}/SCT_JJ LIBS zconfjj zcom)
  endforeach()
  dbsr_util(sum_hh_jj ${UTIL_DIR}/SCT_JJ LIBS zcom)
  dbsr_util(targ_jj   ${UTIL_DIR}/SCT_JJ LIBS zconfjj zconfls zcom)
  foreach(u add_farm add_stgf add_stgfb adf04 beb_ion delete_data eff_cstrn
            gettdonejk gettdoneLS kma_om oma_omb sec_cascb sec_dif_JK sec_dif_JK_ampl
            sec_dif_LS sec_dif_LS_tdone sec_omn sec_par sec_parb sec_sums sec_top
            sec_totalb tma_tmb tmb_omb kma_phase kma_phase_lsp)
    dbsr_util(${u} ${UTIL_DIR}/SCT_LS LIBS zconfls zcom)
  endforeach()
endif()

# ---------------------------------------------------------------------------
# Shared library for the Python outer region (Coulomb functions, ctypes)
# ---------------------------------------------------------------------------
add_library(pydbsr_coulomb SHARED ${CMAKE_CURRENT_SOURCE_DIR}/fortran/pydbsr_coulomb.f90
                                  ${LIBS_DIR}/ZCOM/FUN_coulfg.f ${LIBS_DIR}/ZCOM/FUN_zcfg.f90)
target_compile_options(pydbsr_coulomb PRIVATE -O2)
set_target_properties(pydbsr_coulomb PROPERTIES Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/mod_coulomb)

install(TARGETS ${DBSR_PROGRAMS} RUNTIME DESTINATION ${DBSR_INSTALL_BINDIR})
install(TARGETS pydbsr_coulomb LIBRARY DESTINATION ${DBSR_INSTALL_LIBDIR}
                               RUNTIME DESTINATION ${DBSR_INSTALL_LIBDIR})
message(STATUS "DBSR programs: ${DBSR_PROGRAMS}")
