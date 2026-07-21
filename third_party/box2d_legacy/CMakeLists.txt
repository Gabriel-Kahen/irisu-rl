cmake_minimum_required(VERSION 3.20)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(irisu_box2d_legacy LANGUAGES CXX)
endif()

set(box2d_legacy_sources
    Source/Collision/b2BroadPhase.cpp
    Source/Collision/b2CollideCircle.cpp
    Source/Collision/b2CollidePoly.cpp
    Source/Collision/b2Distance.cpp
    Source/Collision/b2PairManager.cpp
    Source/Collision/b2Shape.cpp
    Source/Common/b2BlockAllocator.cpp
    Source/Common/b2Settings.cpp
    Source/Common/b2StackAllocator.cpp
    Source/Dynamics/Contacts/b2CircleContact.cpp
    Source/Dynamics/Contacts/b2Conservative.cpp
    Source/Dynamics/Contacts/b2Contact.cpp
    Source/Dynamics/Contacts/b2ContactSolver.cpp
    Source/Dynamics/Contacts/b2PolyAndCircleContact.cpp
    Source/Dynamics/Contacts/b2PolyContact.cpp
    Source/Dynamics/Joints/b2DistanceJoint.cpp
    Source/Dynamics/Joints/b2GearJoint.cpp
    Source/Dynamics/Joints/b2Joint.cpp
    Source/Dynamics/Joints/b2MouseJoint.cpp
    Source/Dynamics/Joints/b2PrismaticJoint.cpp
    Source/Dynamics/Joints/b2PulleyJoint.cpp
    Source/Dynamics/Joints/b2RevoluteJoint.cpp
    Source/Dynamics/b2Body.cpp
    Source/Dynamics/b2ContactManager.cpp
    Source/Dynamics/b2Island.cpp
    Source/Dynamics/b2World.cpp
    Source/Dynamics/b2WorldCallbacks.cpp
)

add_library(irisu_box2d_legacy STATIC ${box2d_legacy_sources})
add_library(irisu::box2d_legacy ALIAS irisu_box2d_legacy)

target_include_directories(irisu_box2d_legacy SYSTEM PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>"
)

# The VC8-era source relied on transitive C string declarations. Keep the
# upstream files unchanged and supply those declarations from the build.
if(MSVC)
    target_compile_options(irisu_box2d_legacy PUBLIC
        "/FI${CMAKE_CURRENT_SOURCE_DIR}/compat/cstring_compat.h"
    )
else()
    target_compile_options(irisu_box2d_legacy PUBLIC
        -include "${CMAKE_CURRENT_SOURCE_DIR}/compat/cstring_compat.h"
    )
endif()

if(NOT DEFINED IRISU_LEGACY_FP_MODE)
    string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" irisu_system_processor)
    set(IRISU_LEGACY_FP_MODE "compiler-default")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
       irisu_system_processor MATCHES "^(x86_64|amd64|i.86)$")
        set(IRISU_LEGACY_FP_MODE "x87")
    endif()
endif()

if(IRISU_LEGACY_FP_MODE STREQUAL "x87")
    # The shipped PE32/MSVC wrapper evaluates the r58 solver on the x87 path.
    # GCC's x86-64 default SSE path differs by one ULP on the first measured
    # crowded-board step and diverges from there.
    target_compile_options(irisu_box2d_legacy PRIVATE -mfpmath=387)
endif()
