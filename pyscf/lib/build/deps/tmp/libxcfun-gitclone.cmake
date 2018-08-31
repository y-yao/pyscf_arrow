if("355f42497a9cd17d16ae91da1f1aaaf93756ae8b" STREQUAL "")
  message(FATAL_ERROR "Tag for git checkout should not be empty.")
endif()

set(run 0)

if("/home/yuanyao/pyscf_my/pyscf/lib/build/deps/src/libxcfun-stamp/libxcfun-gitinfo.txt" IS_NEWER_THAN "/home/yuanyao/pyscf_my/pyscf/lib/build/deps/src/libxcfun-stamp/libxcfun-gitclone-lastrun.txt")
  set(run 1)
endif()

if(NOT run)
  message(STATUS "Avoiding repeated git clone, stamp file is up to date: '/home/yuanyao/pyscf_my/pyscf/lib/build/deps/src/libxcfun-stamp/libxcfun-gitclone-lastrun.txt'")
  return()
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -E remove_directory "/home/yuanyao/pyscf_my/pyscf/lib/build/deps/src/libxcfun"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to remove directory: '/home/yuanyao/pyscf_my/pyscf/lib/build/deps/src/libxcfun'")
endif()

# try the clone 3 times incase there is an odd git clone issue
set(error_code 1)
set(number_of_tries 0)
while(error_code AND number_of_tries LESS 3)
  execute_process(
    COMMAND "/usr/bin/git" clone "https://github.com/dftlibs/xcfun.git" "libxcfun"
    WORKING_DIRECTORY "/home/yuanyao/pyscf_my/pyscf/lib/build/deps/src"
    RESULT_VARIABLE error_code
    )
  math(EXPR number_of_tries "${number_of_tries} + 1")
endwhile()
if(number_of_tries GREATER 1)
  message(STATUS "Had to git clone more than once:
          ${number_of_tries} times.")
endif()
if(error_code)
  message(FATAL_ERROR "Failed to clone repository: 'https://github.com/dftlibs/xcfun.git'")
endif()

execute_process(
  COMMAND "/usr/bin/git" checkout 355f42497a9cd17d16ae91da1f1aaaf93756ae8b
  WORKING_DIRECTORY "/home/yuanyao/pyscf_my/pyscf/lib/build/deps/src/libxcfun"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to checkout tag: '355f42497a9cd17d16ae91da1f1aaaf93756ae8b'")
endif()

execute_process(
  COMMAND "/usr/bin/git" submodule init
  WORKING_DIRECTORY "/home/yuanyao/pyscf_my/pyscf/lib/build/deps/src/libxcfun"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to init submodules in: '/home/yuanyao/pyscf_my/pyscf/lib/build/deps/src/libxcfun'")
endif()

execute_process(
  COMMAND "/usr/bin/git" submodule update --recursive
  WORKING_DIRECTORY "/home/yuanyao/pyscf_my/pyscf/lib/build/deps/src/libxcfun"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to update submodules in: '/home/yuanyao/pyscf_my/pyscf/lib/build/deps/src/libxcfun'")
endif()

# Complete success, update the script-last-run stamp file:
#
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy
    "/home/yuanyao/pyscf_my/pyscf/lib/build/deps/src/libxcfun-stamp/libxcfun-gitinfo.txt"
    "/home/yuanyao/pyscf_my/pyscf/lib/build/deps/src/libxcfun-stamp/libxcfun-gitclone-lastrun.txt"
  WORKING_DIRECTORY "/home/yuanyao/pyscf_my/pyscf/lib/build/deps/src/libxcfun"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to copy script-last-run stamp file: '/home/yuanyao/pyscf_my/pyscf/lib/build/deps/src/libxcfun-stamp/libxcfun-gitclone-lastrun.txt'")
endif()

