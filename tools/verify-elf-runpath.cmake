if(NOT DEFINED IRISU_ELF OR NOT DEFINED IRISU_READELF)
  message(FATAL_ERROR "IRISU_ELF and IRISU_READELF are required")
endif()

execute_process(
  COMMAND "${IRISU_READELF}" -dW "${IRISU_ELF}"
  RESULT_VARIABLE status
  OUTPUT_VARIABLE dynamic_section
  ERROR_VARIABLE error
)
if(NOT status EQUAL 0)
  message(FATAL_ERROR "Cannot inspect ${IRISU_ELF}: ${error}")
endif()

string(REGEX MATCHALL
  "\\((RPATH|RUNPATH)\\)[^\n]*\\[[^]]*\\]"
  runpath_lines "${dynamic_section}")
if(NOT runpath_lines)
  message(FATAL_ERROR "${IRISU_ELF} has no exact-library RPATH or RUNPATH")
endif()

foreach(line IN LISTS runpath_lines)
  string(REGEX REPLACE ".*\\[([^]]*)\\].*" "\\1" runpath "${line}")
  if(runpath STREQUAL "" OR runpath MATCHES "(^|:)(:|$)")
    message(FATAL_ERROR "${IRISU_ELF} has an empty RPATH/RUNPATH component: ${runpath}")
  endif()
  string(REPLACE ":" ";" components "${runpath}")
  foreach(component IN LISTS components)
    if(NOT component MATCHES "^/" AND
       NOT component MATCHES "^\\$ORIGIN(/|$)")
      message(FATAL_ERROR
        "${IRISU_ELF} has a relative RPATH/RUNPATH component: ${component}")
    endif()
  endforeach()
endforeach()
