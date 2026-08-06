#!/bin/bash

#region leptonica
cd /build/leptonica
./autogen.sh --prefix=${PREFIX} --enable-static --disable-shared \
    --disable-programs \
    --without-libopenjpeg \
    --host=${CROSS_PREFIX%-}
./configure --prefix=${PREFIX} --enable-static --disable-shared \
    --disable-programs \
    --without-libopenjpeg \
    --host=${CROSS_PREFIX%-} | log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log "Failed to build leptonica"
    exit 1
fi

make -j$(nproc) && make install

if [ ! -f ${PREFIX}/lib/pkgconfig/lept.pc ]; then
    log "Failed to build leptonica"
    exit 1
fi

if [[ ${TARGET_OS} == "windows" ]]; then
    sed -i 's/^Libs: \(.*\)/Libs: \1 -lws2_32/' ${PREFIX}/lib/pkgconfig/lept.pc
fi

if [[ ${TARGET_OS} != "linux" ]]; then
    echo "Libs.private: -lstdc++ -lz -lm -lsharpyuv -lpng16 -ltiff -lgif -ljpeg -lwebp" >>${PREFIX}/lib/pkgconfig/lept.pc
else
    echo "Libs.private: -lstdc++ -lsharpyuv -lpng16 -ltiff -lgif -ljpeg -lwebp" >>${PREFIX}/lib/pkgconfig/lept.pc
fi

cp ${PREFIX}/lib/pkgconfig/lept.pc ${PREFIX}/lib/pkgconfig/liblept.pc
rm -rf /build/leptonica && cd /build
#endregion

#region libtesseract
cd /build/libtesseract

if [[ ${TARGET_OS} == "windows" && ${ARCH} == "aarch64" ]]; then
    # simddetect.cpp calls getenv() but only includes <numeric> and project
    # headers. libstdc++ happens to drag <cstdlib> in transitively, so the GCC
    # platforms build; libc++ (this target's C++ library) does not, giving
    #   error: use of undeclared identifier 'getenv'; did you mean '_wgetenv'?
    # Include it explicitly rather than relying on a transitive include.
    sed -i '/^#include <numeric>/a #include <cstdlib> // for getenv' src/arch/simddetect.cpp
    grep -q '^#include <cstdlib>' src/arch/simddetect.cpp || { log -a "libtesseract: simddetect.cpp cstdlib patch did not apply"; exit 1; }
fi

if [[ ${TARGET_OS} == "darwin" && ${ARCH} == "x86_64" ]]; then
    sed -i '/#include <filesystem>/d' src/ccutil/ccutil.cpp
    sed -i 's/#include <cstring>/#include <cstring> \n#include <sys\/stat.h> \n#include <unistd.h>/' src/ccutil/ccutil.cpp
    sed -i 's/if (tessdata_prefix != nullptr && !std::filesystem::exists(tessdata_prefix)) {/struct stat buffer;\n    if (tessdata_prefix != nullptr \&\& stat(tessdata_prefix, \&buffer) != 0) {/' src/ccutil/ccutil.cpp
    sed -i 's/std::filesystem::exists(subdir)/stat(subdir.c_str(), \&buffer) == 0/' src/ccutil/ccutil.cpp
    sed -i 's/std::filesystem::path subdir = std::filesystem::path(path) \/ "tessdata";/std::string subdir = std::string(path) + "\\\\tessdata";/' src/ccutil/ccutil.cpp
    sed -i -e '/#include <filesystem>/d' \
        -e 's/#include <memory>/#include <memory>\n#include <dirent.h>\n#include <sys\/stat.h>/' \
        -e '/void addAvailableLanguages(const std::string \&datadir,/,/^}/c\\n void addAvailableLanguages(const std::string \&datadir, std::vector<std::string> *langs) {\n  DIR *dir = opendir(datadir.c_str());\n  if (!dir) return;\n\n  struct dirent *entry;\n  while ((entry = readdir(dir)) != nullptr) {\n    if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) continue;\n\n    std::string fullpath = datadir + "/" + entry->d_name;\n    struct stat statbuf;\n    if (stat(fullpath.c_str(), \&statbuf) != 0) continue;\n\n    if (S_ISDIR(statbuf.st_mode)) {\n      addAvailableLanguages(fullpath, langs);\n    } else {\n      std::string name = entry->d_name;\n      size_t pos = name.rfind(".traineddata");\n      if (pos != std::string::npos && pos == name.length() - 12) {\n        langs->push_back(name.substr(0, pos));\n      }\n    }\n  }\n  closedir(dir);\n}' src/api/baseapi.cpp
fi
./autogen.sh --prefix=${PREFIX} --enable-static --disable-shared \
    --disable-doc \
    --without-archive \
    --disable-openmp \
    --without-curl \
    --with-extra-includes=${PREFIX}/include \
    --with-extra-libraries=${PREFIX}/lib \
    --host=${CROSS_PREFIX%-}
./configure --prefix=${PREFIX} --enable-static --disable-shared \
    --disable-doc \
    --without-archive \
    --disable-openmp \
    --without-curl \
    --with-extra-includes=${PREFIX}/include \
    --with-extra-libraries=${PREFIX}/lib \
    --host=${CROSS_PREFIX%-} | log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log "Failed to build libtesseract"
    exit 1
fi

make -j$(nproc) 2>&1 | log -a || { log -a "libtesseract build failed"; exit 1; }
make install 2>&1 | log -a || { log -a "libtesseract install failed"; exit 1; }

if [ ! -f ${PREFIX}/lib/pkgconfig/tesseract.pc ]; then
    log "Failed to build libtesseract"
    exit 1
fi

if [[ ${TARGET_OS} == "windows" ]]; then
    sed -i 's/^Libs: \(.*\)/Libs: \1 -lws2_32/' ${PREFIX}/lib/pkgconfig/tesseract.pc
fi

if [[ ${TARGET_OS} == "darwin" ]]; then
    echo "Libs.private: -lstdc++ -lz -framework Accelerate -lsharpyuv -lpng16 -ltiff -lgif -ljpeg -lwebp" >>${PREFIX}/lib/pkgconfig/tesseract.pc
elif [[ ${TARGET_OS} != "linux" ]]; then
    echo "Libs.private: -lstdc++ -lz -lm -lsharpyuv -lpng16 -ltiff -lgif -ljpeg -lwebp" >>${PREFIX}/lib/pkgconfig/tesseract.pc
else
    echo "Libs.private: -lstdc++ -lsharpyuv -lpng16 -ltiff -lgif -ljpeg -lwebp" >>${PREFIX}/lib/pkgconfig/tesseract.pc
fi

cp ${PREFIX}/lib/pkgconfig/tesseract.pc ${PREFIX}/lib/pkgconfig/libtesseract.pc
#endregion

add_enable "--enable-libtesseract"

exit 0
