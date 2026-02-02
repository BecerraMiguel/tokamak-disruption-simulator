#!/bin/bash

echo "=== Verificando Dependencias para DREAM ==="
echo ""

# Verificar compiladores
echo "1. Compiladores:"
gcc --version | head -1
g++ --version | head -1
gfortran --version | head -1
cmake --version | head -1
echo ""

# Verificar GSL
echo "2. GNU Scientific Library:"
gsl-config --version
echo ""

# Verificar HDF5
echo "3. HDF5:"
h5dump --version 2>&1 | head -1
echo ""

# Verificar Python
echo "4. Python:"
python3 --version
echo ""

# Verificar paquetes Python
echo "5. Paquetes Python:"
python3 -c "import numpy; print('numpy:', numpy.__version__)"
python3 -c "import scipy; print('scipy:', scipy.__version__)"
python3 -c "import matplotlib; print('matplotlib:', matplotlib.__version__)"
python3 -c "import h5py; print('h5py:', h5py.__version__)"
echo ""

echo "=== Verificación completa ==="
