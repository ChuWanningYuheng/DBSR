# Obtain O. Zatsarinny's sources (pinned commits) and make a patched copy
# in the build tree.  The original tree is never modified.

set(DBSR_UPSTREAM_REPOS DBSR3 LIBRARIES UTILS)
set(DBSR_COMMIT_DBSR3     e80b798917783956aef3c9b2df6f7d3e6c34fbcd)
set(DBSR_COMMIT_LIBRARIES fe35c2e2b8b01ffa39e6ef27750a69b93080c4c3)
set(DBSR_COMMIT_UTILS     7c503d4e70c9f254daa304d6795eb7bcea2a4fc2)
# upstream repo names on GitHub (DBSR3 is published as zatsaroi/DBSR3)
set(DBSR_GH_DBSR3     DBSR3)
set(DBSR_GH_LIBRARIES LIBRARIES)
set(DBSR_GH_UTILS     UTILS)

set(DBSR_SRC ${CMAKE_BINARY_DIR}/upstream)
set(_dl ${CMAKE_BINARY_DIR}/_download)

if(NOT DBSR_UPSTREAM_DIR AND DEFINED ENV{DBSR_UPSTREAM_DIR})
  set(DBSR_UPSTREAM_DIR $ENV{DBSR_UPSTREAM_DIR})
endif()
if(NOT DBSR_UPSTREAM_DIR AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/upstream/DBSR3)
  set(DBSR_UPSTREAM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/upstream)   # sdist / conda
endif()

# re-create the patched copy whenever the patch script or the pinned commits change
file(SHA256 ${CMAKE_CURRENT_LIST_DIR}/../tools/patch_upstream.py _patch_hash)
set(_stamp "${_patch_hash} ${DBSR_COMMIT_DBSR3} ${DBSR_COMMIT_LIBRARIES} ${DBSR_COMMIT_UTILS} ${DBSR_UPSTREAM_DIR}")
set(_old_stamp "")
if(EXISTS ${DBSR_SRC}/.patched)
  file(READ ${DBSR_SRC}/.patched _old_stamp)
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_CURRENT_LIST_DIR}/../tools/patch_upstream.py)

if(NOT _old_stamp STREQUAL _stamp)
  file(REMOVE_RECURSE ${DBSR_SRC})
  file(MAKE_DIRECTORY ${DBSR_SRC})
  foreach(repo ${DBSR_UPSTREAM_REPOS})
    if(DBSR_UPSTREAM_DIR)
      message(STATUS "DBSR: using local ${DBSR_UPSTREAM_DIR}/${repo}")
      # copy the real directory (never a symlink: the copy is patched in place)
      get_filename_component(_real ${DBSR_UPSTREAM_DIR}/${repo} REALPATH)
      file(COPY ${_real}/ DESTINATION ${DBSR_SRC}/${repo} FOLLOW_SYMLINK_CHAIN
           PATTERN ".git" EXCLUDE PATTERN "*.pdf" EXCLUDE PATTERN "*.doc*" EXCLUDE
           PATTERN "*_example" EXCLUDE PATTERN "*.bsw" EXCLUDE)
    else()
      set(sha ${DBSR_COMMIT_${repo}})
      set(gh https://github.com/zatsaroi/${DBSR_GH_${repo}})
      set(dest ${DBSR_SRC}/${repo})
      set(ok FALSE)
      find_package(Git QUIET)
      if(GIT_FOUND)                     # git: works behind proxies that block archives
        message(STATUS "DBSR: fetching ${gh} @ ${sha}")
        file(MAKE_DIRECTORY ${dest})
        execute_process(COMMAND ${GIT_EXECUTABLE} init -q WORKING_DIRECTORY ${dest} RESULT_VARIABLE r1)
        execute_process(COMMAND ${GIT_EXECUTABLE} fetch -q --depth 1 ${gh}.git ${sha}
                        WORKING_DIRECTORY ${dest} RESULT_VARIABLE r2)
        if(r1 EQUAL 0 AND r2 EQUAL 0)
          execute_process(COMMAND ${GIT_EXECUTABLE} -c advice.detachedHead=false checkout -q FETCH_HEAD
                          WORKING_DIRECTORY ${dest} RESULT_VARIABLE r3)
          if(r3 EQUAL 0)
            file(REMOVE_RECURSE ${dest}/.git)
            set(ok TRUE)
          endif()
        endif()
        if(NOT ok)
          file(REMOVE_RECURSE ${dest})
        endif()
      endif()
      if(NOT ok)                        # plain HTTPS download of the archive
        set(url ${gh}/archive/${sha}.tar.gz)
        message(STATUS "DBSR: downloading ${url}")
        file(DOWNLOAD ${url} ${_dl}/${repo}.tar.gz STATUS st TLS_VERIFY ON)
        list(GET st 0 code)
        if(NOT code EQUAL 0)
          message(FATAL_ERROR "Could not obtain ${gh} (${sha}): ${st}\n"
            "Set -DDBSR_UPSTREAM_DIR=<dir with DBSR3, LIBRARIES, UTILS> (or the environment "
            "variable DBSR_UPSTREAM_DIR) for offline builds.")
        endif()
        file(ARCHIVE_EXTRACT INPUT ${_dl}/${repo}.tar.gz DESTINATION ${_dl}/x_${repo})
        file(GLOB top ${_dl}/x_${repo}/*)
        file(RENAME ${top} ${dest})
      endif()
    endif()
  endforeach()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  execute_process(COMMAND ${Python3_EXECUTABLE}
                          ${CMAKE_CURRENT_LIST_DIR}/../tools/patch_upstream.py ${DBSR_SRC}
                  RESULT_VARIABLE rc)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "Patching upstream sources failed")
  endif()
  file(WRITE ${DBSR_SRC}/.patched "${_stamp}")
endif()
