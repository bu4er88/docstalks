echo "Installing poppler"
echo ""

git clone https://gitlab.freedesktop.org/poppler/poppler.git
cd poppler
git checkout poppler-0.89.0
brew install pkg-config
mkdir build
cd build
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX:PATH=/usr/local \
    -DENABLE_UNSTABLE_API_ABI_HEADERS=ON \
    -DBUILD_GTK_TESTS=OFF \
    -DBUILD_QT5_TESTS=OFF \
    -DBUILD_CPP_TESTS=OFF \
    -DENABLE_CPP=ON \
    -DENABLE_GLIB=OFF \
    -DENABLE_GOBJECT_INTROSPECTION=OFF \
    -DENABLE_GTK_DOC=OFF \
    -DENABLE_QT5=OFF \
    -DBUILD_SHARED_LIBS=ON \
    ..
make
sudo make install
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig
pip install python-poppler

cd ../../

echo "Installing tesseract"
echo ""

sudo port install autoconf \
                  automake \
                  libtool \
                  pkgconfig \
                  leptonica
sudo port install cairo pango
sudo port install icu +devel
git clone https://github.com/tesseract-ocr/tesseract/
cd tesseract
./autogen.sh
./configure
make training
sudo make install training-install


