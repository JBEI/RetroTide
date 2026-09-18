# RetroTide

This python library designs PKSs automatically from chemical structures.

## Build image:
`docker build -t retrotide .`

## Run container
### Open bash:
`docker run -it --volume "${PWD}:/app" --workdir /app retrotide bash`

### Run jupyter notebooks:
`docker run -it --volume "${PWD}:/app" --workdir /app -p 8888:8888 retrotide jupyter notebook --no-browser --ip=0.0.0.0 --allow-root`

### Copyright Notice for RetroTide (all code except krswaps folder)

RetroTide Copyright (c) 2024, The Regents of the University of California,
through Lawrence Berkeley National Laboratory (subject to receipt of any
required approvals from the U.S. Dept. of Energy) and Northwestern
University. All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative 
works, and perform publicly and display publicly, and to permit others to do so.

### Copyright Notice for KR Swaps (located in krswaps folder)

KR Swaps: A stereochemistry correction algorithm for computer-aided chimeric type I polyketide synthase design (KR Swaps) Copyright (c) 2026, Northwestern University.  All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative 
works, and perform publicly and display publicly, and to permit others to do so.

### License and redistribution

The RetroTide license is located in license.txt, and the KR Swaps license is located in krswaps/license.txt
