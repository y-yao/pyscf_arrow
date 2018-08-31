FILE(REMOVE_RECURSE
  "CMakeFiles/libxcfun"
  "CMakeFiles/libxcfun-complete"
  "deps/src/libxcfun-stamp/libxcfun-install"
  "deps/src/libxcfun-stamp/libxcfun-mkdir"
  "deps/src/libxcfun-stamp/libxcfun-download"
  "deps/src/libxcfun-stamp/libxcfun-update"
  "deps/src/libxcfun-stamp/libxcfun-patch"
  "deps/src/libxcfun-stamp/libxcfun-configure"
  "deps/src/libxcfun-stamp/libxcfun-build"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/libxcfun.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
