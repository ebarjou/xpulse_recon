Clone
git clone https://github.com/ebarjou/xpulse_recon
cd xpulse_recon
git submodule update --init --recursive

Build (MSVC)
mkdir build
cd build
cmake ..
MSBuild.exe .\ALL_BUILD.vcxproj /property:Configuration=Release