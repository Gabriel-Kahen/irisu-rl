# GNU tar 1.34's host compatibility shims collide with the newer five-argument
# ACL API on current Arch build hosts. The guest does not use host tar ACL or
# xattr support; disabling those optional features keeps the pinned Buildroot
# release buildable without changing any target package.
HOST_TAR_CONF_OPTS += --without-posix-acls --without-xattrs

# GNU cpio 2.15 still uses pre-C23 declarations. Modern host compilers default
# to C23, so constrain this host-only build without changing target binaries or
# the C++ mode used to bootstrap GCC.
HOST_CPIO_CONF_ENV += CFLAGS="$(HOST_CFLAGS) -std=gnu17"

# GCC 12's bundled libcody assumes the pre-C++20 type of u8 literals. The
# top-level recursive builds otherwise override the C++11 mode selected by
# libcody's configure script when bootstrapping with a newer host compiler.
HOST_GCC_INITIAL_MAKE_OPTS += CXX="$(HOSTCXX) -std=gnu++11"
HOST_GCC_FINAL_MAKE_OPTS += CXX="$(HOSTCXX) -std=gnu++11"
HOST_GCC_INITIAL_CONF_ENV += CXX="$(HOSTCXX) -std=gnu++11"
HOST_GCC_FINAL_CONF_ENV += CXX="$(HOSTCXX) -std=gnu++11"
