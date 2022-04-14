#!/bin/bash

# Install singularityCE dependencies
apt-get update && apt-get install -y \
build-essential \
libssl-dev \
uuid-dev \
libgpgme11-dev \
squashfs-tools \
libseccomp-dev \
wget \
pkg-config \
git \
libz-dev \
cryptsetup

# Install squashfs-tools-4.5
### This is a fix for a `liblzma.so.5` issue on Ubuntu systems.
REQUIRED_PKG="mksquashfs"
VERSION=4.5
PKG_OK=$($REQUIRED_PKG -version | cut -d' ' -f3 | grep $VERSION)
echo Checking for $REQUIRED_PKG: $PKG_OK
if [ "" = "$PKG_OK" ]; then
  echo "Installing $REQUIRED_PKG."
  wget https://downloads.sourceforge.net/project/squashfs/squashfs/squashfs4.5/squashfs4.5.tar.gz
  tar -zxvf squashfs4.5.tar.gz
  (
    cd squashfs-tools-4.5/squashfs-tools
    make && make install
  )
  rm -rf squashfs*
fi

### Install and configure go (https://golang.org/doc/install)
REQUIRED_PKG="go"
VERSION=1.17.7
PKG_OK=$(go version | cut -d' ' -f3 | grep $VERSION)
echo Checking for $REQUIRED_PKG: $PKG_OK
if [ "" = "$PKG_OK" ]; then
  echo "Installing $REQUIRED_PKG."
  wget https://go.dev/dl/go1.17.7.linux-amd64.tar.gz
  rm -rf /usr/local/go
  tar -C /usr/local -xzf go1.17.7.linux-amd64.tar.gz
  export PATH=$PATH:/usr/local/go/bin
fi


# Install singularityCE (latest tested: 3.8.3)
REQUIRED_PKG="singularity"
export VERSION=3.8.3
PKG_OK=$($REQUIRED_PKG --version | cut -d' ' -f3 | grep $VERSION)
echo Checking for $REQUIRED_PKG: $PKG_OK
if [ "" = "$PKG_OK" ]; then
  echo "Installing $REQUIRED_PKG."
  wget https://github.com/sylabs/singularity/releases/download/v${VERSION}/singularity-ce-${VERSION}.tar.gz
  tar -xzf singularity-ce-${VERSION}.tar.gz
  ### Build the code
  (
    cd singularity-ce-${VERSION}
    ./mconfig
    make -C builddir
    make -C builddir install
  )
  rm -rf singularity-ce-${VERSION}
fi