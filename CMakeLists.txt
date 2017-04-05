################################################################################
#
# Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO USER:   
#
# This source code is subject to NVIDIA ownership rights under U.S. and 
# international Copyright laws.  
#
# NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
# CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
# IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
# REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
# MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
# IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
# OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
# OR PERFORMANCE OF THIS SOURCE CODE.  
#
# U.S. Government End Users.  This source code is a "commercial item" as 
# that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
# "commercial computer software" and "commercial computer software 
# documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
# and is provided to the U.S. Government only as a commercial end item.  
# Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
# 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
# source code with only those rights set forth herein.
#
################################################################################

cmake_minimum_required(VERSION 2.8)


project(sdf)

#Determine Version information
#Specify version information
set (MAJOR_VERSION 0)
set (MINOR_VERSION 2)
set (PATCH_VERSION 0)
set (TWEAK_VERSION 0)





#If CPACK is installed, make it an option
find_program(CPACK_INSTALLED cpack DOC "CPACK executable")
if( CPACK_INSTALLED )
   option( MAKE_DEB_PACKAGE "Make a Debian package" OFF )
endif( CPACK_INSTALLED )



##########################################333
# Build section
##########################################333
set(DEST_DIR "$ENV{PWD}/INSTALL")
#set(CMAKE_INSTALL_RPATH ${LIBNVVM_HOME})

set(CONFIG_DIR "${DEST_DIR}/etc/aqueti")

if(CMAKE_SIZEOF_VOID_P STREQUAL "8")
  add_definitions(-DBUILD_64_BIT)
endif()

set(CMAKE_INCLUDE_CURRENT_DIR YES)

add_executable(simple simple.c)

set(CUDA_LIB cuda)
set(NVVM_LIB nvvm)
target_link_libraries(simple ${NVVM_LIB} ${CUDA_LIB})

if (WIN32)
  set_target_properties(simple PROPERTIES
    COMPILE_FLAGS "/wd4996")
else (WIN32)
  set_target_properties(simple PROPERTIES
    LINK_FLAGS    "-Wl,-rpath,${LIBNVVM_RPATH}")
endif (WIN32)


#############################################
#Package stuff
#############################################
if( MAKE_DEB_PACKAGE )
      set(CPACK_PACKAGE_NAME ${CMAKE_PROJECT_NAME})
      set(CPACK_PACKAGE_VERSION "${MAJOR_VERSION}.${MINOR_VERSION}.${PATCH_VERSION}.${TWEAK_VERSION}")
      set(CPACK_PACKAGING_INSTALL_PREFIX "/")
      set(CPACK_PACKAGE_ARCH "linux64")
      set(CPACK_PACKAGE_VENDOR "Aqueti")
      set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Aqueti Inc.")
      set(CPACK_DEBIAN_PACKAGE_DESCRIPTION "Aqueti Tool Library")
      set(CPACK_DEBIAN_PACKAGE_DEPENDS "libcudart7.5 (>=7.5.18)")
      set(CPACK_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}-${CPACK_PACKAGE_ARCH}")
      set( PACKAGE_FILE "${CMAKE_CURRENT_BINARY_DIR}/${CPACK_PACKAGE_FILE_NAME}.deb")
      set( CPACK_GENERATOR "DEB")

      INCLUDE(CPack)

      #The following is the command sequence to generate the package and install it in the
      #CMAKE_INSTALL_PREFIX/deb directory. The first command executes a "make package" the 
      #second creates the destination directory, and the third copies the debian file."
      set(cmd  "make")
      set(arg1 "package")
      set(mkdir "mkdir")
      set(mkdirparm "-p")
      set(cpcmd "cp")
      set(dest "${CMAKE_INSTALL_PREFIX}/deb")

      #This custom command will build the debian file as the POST_BUILD step of the atl
      add_custom_command(TARGET simple POST_BUILD
         COMMAND ${cmd} ${arg1}
         COMMAND ${mkdir} ${mkdirparm} ${dest} 
         COMMAND ${cpcmd} ${PACKAGE_FILE} ${dest}
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
         COMMENT  "Make Package"
      )
endif()

install(TARGETS 
        simple 
        DESTINATION bin)
install(FILES simple-gpu.ll
        simple-gpu64.ll
        DESTINATION etc/aqueti)
