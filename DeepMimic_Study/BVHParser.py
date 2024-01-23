import numpy as np
import math as m
import argparse
import glm

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--bvh", dest="bvh", action="store")
args = parser.parse_args()

def Rx(theta):
    return np.matrix([[ 1, 0           , 0           , 0],
                      [ 0, m.cos(theta),-m.sin(theta), 0],
                      [ 0, m.sin(theta), m.cos(theta), 0],
                      [ 0,            0,            0, 1]])
  
def Ry(theta):
    return np.matrix([[ m.cos(theta), 0, m.sin(theta), 0],
                      [ 0           , 1, 0           , 0],
                      [-m.sin(theta), 0, m.cos(theta), 0],
                      [ 0,            0,            0, 1]])
  
def Rz(theta):
    return np.matrix([[ m.cos(theta), -m.sin(theta), 0, 0],
                      [ m.sin(theta),  m.cos(theta), 0, 0],
                      [ 0,            0,             1, 0],
                      [ 0,            0,             0, 1]])
def Tx(d):
    return np.matrix([[1, 0, 0, d],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

def Ty(d):
    return np.matrix([[1, 0, 0, 0],
                      [0, 1, 0, d],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

def Tz(d):
    return np.matrix([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, d],
                      [0, 0, 0, 1]])
def Txyz(offset):
    return np.matrix([[1, 0, 0, offset[0]],
                      [0, 1, 0, offset[1]],
                      [0, 0, 1, offset[2]],
                      [0, 0, 0,        1]])

class Joint:
    def __init__(self):
        self.name = ""
        self.offset = glm.vec3(0)
        self.channelnum = 0
        self.channels = []
        self.indices = []
        self.children = []
        self.depth = 0
        self.parent = None

    def print(self):
        print("Name:", self.name)
        print("Offset:", self.offset)
        print("# of channels:", self.channelnum)
        print("Channels:", self.channels)
        print("Children:", end=" ")
        for child in self.children:
            print(child.name, end=" ")
        print()
        print("Depth:", self.depth)
        if self.parent != None:
            print("Parent:", self.parent.name)
        print()

        for child in self.children:
            child.print()

class BVH:
    def __init__(self):
        self.names = []
        self.offsets = []
        self.channelnum = []
        self.channels = []
        self.parents = []
        self.nums = []
        self.indices = []

        self.frames = 0
        self.frametime = 0
        self.framelist = []

        self.pointclouds = []

        self.root = None

def BVHParse(filename):
    if filename[-3:] == "bvh":
        resultBVH = BVH()

        nowjoint = Joint()
        parentjoint = None

        f = open(filename, 'r')

        depth = 0
        num = 0
        parent = -1
        now = 0
        num = 0
        index = 0

        names = []
        offsets = []
        channelnum = []
        channels = []
        parents = []
        nums = []
        indices = []

        # first line is HIERARCHY
        line = f.readline()

        # Parsing Hierarchy
        while True: 
            line = f.readline()

            if line.strip() == "{":
                depth += 1
                parent = now
                #print("{ parent", parent, "now", now, "num", num)

            elif line.strip() == "}":
                depth -= 1
                now = parent
                parent = parents[now]
                #print("} parent", parent, "now", now, "num", num)

                nowjoint = parentjoint
                parentjoint = nowjoint.parent
            else:
                check = line.split()
                if check[0] == "ROOT" or check[0] =="JOINT" or check[0] == "End":
                    parentjoint = nowjoint
                    nowjoint = Joint()
                    parentjoint.children.append(nowjoint)
                    nowjoint.depth = depth

                    if check[0] == "End":
                        names.append("End Site")
                        channelnum.append(0)
                        channels.append([])
                        indices.append([])
                        #print("End Site")

                        nowjoint.name = "End Site"
                    else:
                        names.append(check[1])
                        #print(check[1])

                        nowjoint.name = check[1]

                    parents.append(parent)
                    nums.append(num)
                    now = num
                    num += 1
                    #print("parents", parents, "nums", nums, "parent", parent, "now", now)

                    nowjoint.parent = parentjoint

                    if check[0] == "ROOT": continue

                elif check[0] == "OFFSET":
                    check.pop(0) # OFFSET
                    #offsets.append(np.float_(check))
                    offsets.append(glm.vec3(float(check[0]), float(check[1]), float(check[2])))

                    nowjoint.offset = glm.vec3(float(check[0]), float(check[1]), float(check[2]))

                elif check[0] == "CHANNELS":
                    check.pop(0) # CHANNELS
                    n = int(check[0])
                    channelnum.append(n)
                    indexlist = []
                    for i in range(n):
                        indexlist.append(index)
                        index += 1
                    indices.append(indexlist)
                    check.pop(0) # Num of Channels
                    channels.append(check)

                    nowjoint.channelnum = n
                    nowjoint.indices = indexlist
                    nowjoint.channels = check

            if depth == 0:
                break

        # MOTION
        line = f.readline()

        line = f.readline().split()
        frames = int(line[1])
        line = f.readline().split()
        frametime = float(line[2])
        framelist = []

        for i in range(frames):
            framelist.append(np.float_(f.readline().split()))

        jointnum = len(nums)
        f.close()

        # print(names)
        # print(offsets)
        # print(channelnum)
        # print(channels)
        # print(parents)
        # print(nums)
        # print(indices)

        # print(frames, frametime, len(framelist))

        resultBVH.names = names
        resultBVH.offsets = offsets
        resultBVH.channelnum = channelnum
        resultBVH.channels = channels
        resultBVH.parents = parents
        resultBVH.nums = nums
        resultBVH.indices = indices
        
        resultBVH.frames = frames
        resultBVH.frametime = frametime
        resultBVH.framelist = framelist

        #nowjoint.children[0].print()

        resultBVH.root = nowjoint.children[0]

        pointclouds = []

        for i in range(frames):
            Matrices = []
            for j in range(jointnum):
                Transform = np.identity(4)
                for k in range(channelnum[j]):
                    cn = channels[j][k]
                    mulmat = np.zeros(4)
                    nowindex = indices[j][k]
                    if cn == "Xposition":
                        mulmat = Tx(framelist[i][nowindex])
                    elif cn == "Yposition":
                        mulmat = Ty(framelist[i][nowindex])
                    elif cn == "Zposition":
                        mulmat = Tz(framelist[i][nowindex])
                    elif cn == "Zrotation":
                        mulmat = Rz(np.radians(framelist[i][nowindex]))
                    elif cn == "Xrotation":
                        mulmat = Rx(np.radians(framelist[i][nowindex]))
                    elif cn == "Yrotation":
                        mulmat = Ry(np.radians(framelist[i][nowindex]))

                    Transform = np.matmul(Transform, mulmat)
                    #Transform = np.matmul(mulmat, Transform)
                Matrices.append(Transform)

            xs = []
            ys = []
            zs = []
            pointcloud = []
            for j in range(jointnum):
                parentMat = np.identity(4)
                if j != 0:
                    parentMat = Matrices[parents[j]]
                nowMat= Matrices[j]

                Matrices[j] = np.matmul(parentMat, Txyz(offsets[j]))
                Matrices[j] = np.matmul(Matrices[j], nowMat)

                x = Matrices[j][0,3]
                y = Matrices[j][1,3]
                z = Matrices[j][2,3]
                xs.append(x)
                zs.append(y)
                ys.append(z)
                pointcloud.append(glm.vec3(x, y, z))

            pointclouds.append(pointcloud)

        resultBVH.pointclouds = pointclouds

        return resultBVH

if __name__ == "__main__":
    result = BVHParse(args.bvh)