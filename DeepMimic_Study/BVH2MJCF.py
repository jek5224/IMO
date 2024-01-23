from BVHParser import *

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--bvh", dest="bvh", action="store")
args = parser.parse_args()

result = BVHParse(args.bvh)

#result.root.print()

scale = 1

def tw(f, string, num): # tabwrite
    for i in range(num):
        f.write('  ')
    f.write(string + '\n')

def bw(f, joint, n):    # bodywrite
    if joint.name == "End Site":
        return
    
    if joint.depth == 0:
        global scale

        minlen = 10000000
        for child in joint.children:
            candlen = glm.length(child.offset)
            if candlen < minlen:
                minlen = candlen

        scale = 0.09 / minlen

        tw(f, '<body name="' + joint.name + '" pos="0 0 0" childclass="body">', n)
        tw(f, '<freejoint name="' + joint.name + '"/>', n + 1)
        tw(f, '<site name="' + joint.name + '" class="force-torque"/>', n + 1)
        tw(f, '<geom name="' + joint.name + '" type="sphere" pos="0 0 0" size=".09" density="2226"/>', n + 1)
        tw(f, '<site name="' + joint.name + '" class="touch" type="sphere" pos="0 0 0" size="0.091"/>', n + 1)
    else:
        string = '<body name="' + joint.name + '" pos="'
        string += str(np.round(joint.offset[0], 6)) + ' '
        string += str(np.round(joint.offset[1], 6)) + ' '
        string += str(np.round(joint.offset[2], 6)) + '">'
        tw(f, string, n)

        tw(f, '<joint name="' + joint.name + '_x" pos="0 0 0" axis="1 0 0" range="-90 90"' +\
           'stiffness="1000" damping="100" armature=".02"/>', n + 1)
        tw(f, '<joint name="' + joint.name + '_y" pos="0 0 0" axis="0 1 0" range="-90 90"' +\
           'stiffness="1000" damping="100" armature=".02"/>', n + 1)
        tw(f, '<joint name="' + joint.name + '_z" pos="0 0 0" axis="0 0 1" range="-90 90"' +\
           'stiffness="1000" damping="100" armature=".02"/>', n + 1)
        
    for child in joint.children:
        bw(f, child, n + 1)

    tw(f, '</body>', n)

f = open(args.bvh[:-4] + '.xml', 'w')

tw(f, '<mujoco>', 0)
tw(f, '<default>', 1)
tw(f, '<motor ctrlrange="-1 1" ctrllimited="true"/>', 2)
tw(f, '<default class="body">', 2)
tw(f, '<geom type="capsule" condim="1" friction="1.0 0.05 0.05" solimp=".9 .99 .003" solref=".015 1"/>', 3)
tw(f, '<joint type="hinge" damping="0.1" stiffness="5" armature=".007" limited="true" solimplimit="0 .99 .01"/>', 3)
tw(f, '<site size=".04" group="3"/>', 3)
tw(f, '<default class="force-torque">', 3)
tw(f, '<site type="box" size=".01 .01 .02" rgba="1 0 0 1" />', 4)
tw(f, '</default>', 3)
tw(f, '<default class="touch">', 3)
tw(f, '<site type="capsule" rgba="0 0 1 .3"/>', 4)
tw(f, '</default>', 3)
tw(f, '</default>', 2) 
tw(f, '</default>', 1)

tw(f, '<worldbody>', 1)
bw(f, result.root, 1)
tw(f, '</worldbody>', 1)

tw(f, '</mujoco>', 0)
f.close()