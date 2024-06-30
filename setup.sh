# Docker install
echo "Installing Docker"
echo ""
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

# Install poppler
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

# Install tesseract
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

# Install python venv module 
apt install python3.12-venv
python3 -m venv venv
source venv/bin/activate

# Install packages
pip3 install -r requirements-prod.txt

