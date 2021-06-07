import numpy as np
import math
import cupy.cuda.nvtx as nvtx
from MDAnalysis.lib.formats.libdcd import DCDFile
from timeit import default_timer as timer
from numba import  njit

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
    inconf = 10
    nbin   = 2000
    global xbox, ybox, zbox
    file   = "input/alk.traj.dcd"
    infile = DCDFile(file)
    pairfile = open("RDF.dat", "w+")
    stwo     = open("Pair_entropy.dat", "w+")

    numatm, nconf = dcdreadhead(infile)
    print("Dcd file has {} atoms and {} frames".format(numatm, nconf))
    if inconf > nconf:
        print("nconf is reset to {}".format(nconf))
    else:
        nconf = inconf
    print("Calculating RDF for {} frames".format(nconf))
    #numatm = 50
    sizef =  nconf * numatm
    sizebin = nbin
    ########### reading cordinates ##############
    nvtx.RangePush("Read_File")
    xbox, ybox, zbox, h_x, h_y, h_z = dcdreadframe(infile, numatm, nconf)
    nvtx.RangePop() # pop for reading file

    h_g2 = np.zeros(sizebin, dtype=np.longlong)
    print("Reading of input file is completed")
    ############# This where we will concentrate #########################
    nvtx.RangePush("Pair_Circulation")
    h_g2 = pair_gpu(h_x, h_y, h_z, h_g2, numatm, nconf, xbox, ybox, zbox, nbin)
    nvtx.RangePop() #pop for Pair Calculation
    ######################################################################
    pi = math.acos(np.long(-1.0))
    rho = (numatm) / (xbox * ybox * zbox)
    norm = (np.long(4.0) * pi * rho) / np.long(3.0)
    g2 = np.zeros(nbin, dtype=np.float32)
    s2 = np.long(0.0);
    s2bond = np.long(0.0)
    lngrbond = np.float(0.0)
    box = min(xbox, ybox)
    box = min(box, zbox)
    _del = box / (np.long(2.0) * nbin)
    gr = np.float(0.0)
    # loop to calculate entropy
    nvtx.RangePush("Entropy_Calculation")
    for i in range(nbin):
        rl = (i) * _del
        ru = rl + _del
        nideal = norm * (ru * ru * ru - rl * rl * rl)
        g2[i] = h_g2[i] / (nconf * numatm * nideal)
        r = (i) * _del
        temp = (i + 0.5) * _del
        pairfile.write(str(temp) + " " + str(g2[i]) + "\n")

        if r < np.long(2.0):
            gr = np.long(0.0)
        else:
            gr = g2[i]
        if gr < 1e-5:
            lngr = np.long(0.0)
        else:
            lngr = math.log(gr)
        if g2[i] < 1e-6:
            lngrbond = np.long(0.0)
        else:
            lngrbond = math.log(g2[i])
        s2 = s2 - (np.long(2.0) * pi * rho * ((gr * lngr) - gr + np.long(1.0)) * _del * r * r)
        s2bond = s2bond - np.long(2.0) * pi * rho * ((g2[i] * lngrbond) - g2[i] + np.long(1.0)) * _del * r * r

    nvtx.RangePop() # pop for entropy Calculation
    stwo.writelines("s2 value is {}\n".format(s2))
    stwo.writelines("s2bond value is {}".format(s2bond))

    print("\n s2 value is {}\n".format(s2))
    print("s2bond value is {}\n".format(s2bond))
    print("#Freeing Host memory")
    del(h_x)
    del(h_y)
    del(h_z)
    del(h_g2)
    print("#Number of atoms processed: {}  \n".format(numatm))
    print("#number of confs processed: {} \n".format(nconf))
    total_time = timer() - start
    print("total time spent:", total_time)

@njit()
def pair_gpu(d_x, d_y, d_z, d_g2, numatm, nconf, xbox, ybox, zbox, d_bin):
    box = min(xbox, ybox)
    box = min(box, zbox)
    _del = box / (2.0 * d_bin)
    cut = box * 0.5
    #print("\n {} {}".format(nconf, numatm))

    for frame in range(nconf):
        #print("\n {}".format(frame))
        for id1 in range(numatm):
            for id2 in range(numatm):
                dx = d_x[frame * numatm + id1] - d_x[frame * numatm + id2]
                dy = d_y[frame * numatm + id1] - d_y[frame * numatm + id2]
                dz = d_z[frame * numatm + id1] - d_z[frame * numatm + id2 ]
                dx = dx - xbox * (round(dx / xbox))
                dy = dy - ybox * (round(dy / ybox))
                dz = dz - zbox * (round(dz / zbox))

                r = math.sqrt(dx * dx + dy * dy + dz * dz)
                if r < cut :
                    ig2  = int((r/_del))
                    d_g2[ig2] = d_g2[ig2] + 1

    return d_g2


if __name__ == "__main__":
    main()