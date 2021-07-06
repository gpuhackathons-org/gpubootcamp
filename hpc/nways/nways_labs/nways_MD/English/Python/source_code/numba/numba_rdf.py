# Copyright (c) 2021 NVIDIA Corporation. All rights reserved.

import numpy as np
import math
import cupy.cuda.nvtx as nvtx
from MDAnalysis.lib.formats.libdcd import DCDFile
from timeit import default_timer as timer
import numba.cuda as cuda
import os
from pathlib import Path

def dcdreadhead(infile):
    nconf   = infile.n_frames
    _infile = infile.header
    numatm  = _infile['natoms']
    return numatm, nconf

def dcdreadframe(infile, numatm, nconf):

    d_x = np.zeros(numatm * nconf, dtype=np.float64)
    d_y = np.zeros(numatm * nconf, dtype=np.float64)
    d_z = np.zeros(numatm * nconf, dtype=np.float64)

    for i in range(nconf):
        data = infile.readframes(i, i+1)
        box = data[1]
        atomset = data[0][0]
        xbox = round(box[0][0], 8)
        ybox = round(box[0][2],8)
        zbox = round(box[0][5], 8)

        for row in range(numatm):
            d_x[i * numatm + row] = round(atomset[row][0], 8) # 0 is column
            d_y[i * numatm + row] = round(atomset[row][1], 8)  # 1 is column
            d_z[i * numatm + row] = round(atomset[row][2], 8)  # 2 is column

    return xbox, ybox, zbox, d_x, d_y, d_z

def main():
    start = timer()
    ########## Input Details ###########
    global xbox, ybox, zbox
    inconf = 10
    nbin   =np.int32(2000)
    xbox   = np.float32(0)
    ybox   =np.float32(0)
    zbox   = np.float32(0)

    ########use on jupyter notebook#######
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    dataRoot = Path(fileDir).parents[1]
    file = os.path.join(dataRoot, 'source_code/input/alk.traj.dcd')

    ########use on local computer##########
    # file   = "input/alk.traj.dcd"
    #######################################

    infile = DCDFile(file)

    pairfile = open("numba_RDF.dat", "w+")
    stwo = open("numba_Pair_entropy.dat", "w+")

    numatm, nconf = dcdreadhead(infile)

    print("Dcd file has {} atoms and {} frames".format(numatm, nconf))
    if inconf > nconf:
        print("nconf is reset to {}".format(nconf))
    else:
        nconf = inconf
    print("Calculating RDF for {} frames".format(nconf))

    #numatm = 100
    sizef =  nconf * numatm
    sizebin = nbin

    ########### reading cordinates ##############
    nvtx.RangePush("Read_File")
    xbox, ybox, zbox, d_x, d_y, d_z = dcdreadframe(infile, numatm, nconf)
    nvtx.RangePop()  # pop for reading file
    print("Reading of input file is completed")

    ############################## Numba KERNEL #################################################

    nthreads = 128;
    near2 = nthreads * (int(0.5 * numatm * (numatm - 1) / nthreads) + 1);
    nblock = (near2 / nthreads);
    print(" Initial blocks are {} and now changing to".format(nblock))
    maxblock = 65535
    blockloop = int(nblock / maxblock)
    if blockloop != 0:
        nblock = maxblock
    print("{} and will run over {} blockloops".format(nblock, blockloop+1))

    # cp.cuda.runtime.memset(d_g2,0,sizebin)
    d_g2 = np.zeros(sizebin, dtype=np.int64)
    d_g2 = cuda.to_device(d_g2) #numba copy to device

    nvtx.RangePush("Pair_Circulation_Numba")
    t1 = timer()
    for bl in range(blockloop+1):
        pair_gpu_kernel[nblock,nthreads ](d_x, d_y, d_z, d_g2, numatm, nconf, xbox, ybox, zbox, nbin, bl)  ## numba jit kernel
    cuda.synchronize()
    print("Kernel compute time:", timer() - t1)
    d_g2  = d_g2.copy_to_host() ## numba copy to host
    nvtx.RangePop()  # pop for Pair Calculation

    pi = math.acos(np.int64(-1.0))
    rho = (numatm) / (xbox * ybox * zbox)
    norm = (np.int64(4.0) * pi * rho) / np.int64(3.0)
    g2 = np.zeros(nbin, dtype=np.float32)
    s2 =np.int64(0.0); s2bond = np.int64(0.0)
    lngrbond = np.float32(0.0)
    box = min(xbox, ybox)
    box = min(box, zbox)
    _del =box / (np.int64(2.0) * nbin)
    gr = np.float32(0.0)
    # loop to calculate entropy
    nvtx.RangePush("Entropy_Calculation")
    for i in range(nbin):
        rl = (i) * _del
        ru = rl + _del
        nideal = norm * (ru * ru * ru - rl * rl * rl)
        g2[i] = d_g2[i] / (nconf * numatm * nideal)
        r = (i) * _del
        temp = (i + 0.5) * _del
        pairfile.write(str(temp) + " " + str(g2[i]) + "\n")

        if r < np.int64(2.0):
            gr = np.int64(0.0)
        else:
            gr = g2[i]
        if gr < 1e-5:
            lngr = np.int64(0.0)
        else:
            lngr = math.log(gr)
        if g2[i] < 1e-6:
            lngrbond = np.int64(0.0)
        else:
            lngrbond = math.log(g2[i])
        s2 = s2 - (np.int64(2.0) * pi * rho * ((gr * lngr) - gr + np.int64(1.0)) * _del * r * r)
        s2bond = s2bond - np.int64(2.0) * pi * rho * ((g2[i] * lngrbond) - g2[i] + np.int64(1.0)) * _del * r * r

    nvtx.RangePop()  # pop for entropy Calculation
    stwo.writelines("s2 value is {}\n".format(s2))
    stwo.writelines("s2bond value is {}".format(s2bond))

    print("#Freeing Host memory")
    del (d_x)
    del (d_y)
    del (d_z)
    del (d_g2)
    print("#Number of atoms processed: {}  \n".format(numatm))
    print("#number of confs processed: {} \n".format(nconf))
    total_time = timer() - start
    print("total time spent:", total_time)


##--------------------------NUMBA KERNEL---------------------------------------------------------##
@cuda.jit
def pair_gpu_kernel(d_x, d_y,d_z, d_g2, numatm, nconf, xbox, ybox,zbox,d_bin, bl):
    box = min(xbox, ybox)
    box = min(box, zbox)
    _del= box / (2.0 * d_bin)
    cut = box * 0.5;
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    maxi = min(int(0.5 * numatm * (numatm - 1) - (bl * 65535 * 128)), (65535 * 128))

    if i < maxi:
        thisi=bl * 65535 * 128+i
        n = (0.5) * (1+ ( math.sqrt (1.0+4.0 * 2.0 * thisi)))
        id1 = int(n)
        id2 = thisi-(0.5 * id1 * (id1-1))
        for frame in range(0, nconf):
            t1 = int(frame * numatm+id1)
            t2 = int(frame * numatm+id2)
            dx = d_x[t1] - d_x[t2]
            dy = d_y[t1] - d_y[t2]
            dz = d_z[t1] - d_z[t2]
            dx = dx - xbox * (round(dx / xbox))
            dy = dy - ybox * (round(dy / ybox))
            dz = dz - zbox * (round(dz / zbox))

            r= math.sqrt(dx * dx+dy * dy+dz * dz)
            if r < cut:
                ig2=(int)(r / _del )
                cuda.atomic.add(d_g2, ig2, 2)

##--------------------------END NUMBA KERNEL---------------------------------------------------------##


if __name__ == "__main__":
    main()

