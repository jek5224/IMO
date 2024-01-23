from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import glm
import numpy as np
import math as m
import copy

# https://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf
# In BVH Rotation Order! RzRyRx
def rotation_zyx(matrix):

    r11, r12, r13 = matrix[0][0], matrix[1][0], matrix[2][0]
    r21, r22, r23 = matrix[0][1], matrix[1][1], matrix[2][1]
    r31, r32, r33 = matrix[0][2], matrix[1][2], matrix[2][2]

    t2 = np.arcsin(-r31)

    ct2 = np.cos(t2)

    if ct2 != 0:
        t1 = m.atan2(r21, r11)
        t3 = m.atan2(r32, r33)
    else:   # Singular point
        if t2 == np.pi / 2:
            t1 = 0
            t3 = m.atan2(r12, r22)
        else:
            t1 = 0
            t3 = m.atan2(-r12, r22)

    t1 = t1 * 180 / np.pi
    t2 = t2 * 180 / np.pi
    t3 = t3 * 180 / np.pi

    return t1, t2, t3

width = 1500
height = 800

scale = 100

frames = 0
frametime = 1 / 30

nowframe = 0
framenum = 1

view = glm.normalize(glm.vec3(1.0, 1.0, 1.0))
lookat = glm.vec3(0.0, 0.0, 0.0)
viewangle = 60.0
d = 0.3
fix_d = 5 * scale
rot_angle = 0

posed = True
camera_fix = False

f = open('joints.txt', 'r')

frames = int(f.readline())

# OpenPose Joint Ordering
# For other conventions, this should be changed
edges = [
    [8, 1],
        [1, 0],
            [0, 15],
                [15, 17],
            [0, 16],
                [16, 18],
        [1, 2],
            [2, 3],
                [3, 4],
        [1, 5],
            [5, 6],
                [6, 7],
    [8, 9],
        [9, 10],
            [10, 11],
                [11, 22],
                    [22, 23],
                [11, 24],
    [8, 12],
        [12, 13],
            [13, 14],
                [14, 19],
                    [19, 20],
                [14, 21]
]

# Joint names in number order; 0-nose, 1-neck, ...
OPENPOSE = [
    'nose', 'neck',
    'right_shoulder', 'right_elbow', 'right_wrist',
    'left_shoulder', 'left_elbow', 'left_wrist',
    'pelvis',
    'right_hip', 'right_knee', 'right_ankle',
    'left_hip', 'left_knee', 'left_ankle',
    'right_eye', 'left_eye', 'right_ear', 'left_ear',
    'left_big_toe', 'left_small_toe', 'left_heel',
    'right_big_toe', 'right_small_toe', 'right_heel'
]

posed_list = []     # Save all joint positions
unposed_list = []   # Save T pose joint (rotated) from SMPLify, later modified
camera_list = []    # Save camera positions

for j in range(frames):
    joints = []
    for i in range(25):
        joint = [float(x) * scale for x in f.readline().split()]
        joints.append(glm.vec3(-joint[0], joint[1], joint[2]))

    line = f.readline()
    posed_list.append(joints)

line = f.readline()    # Space
for j in range(frames):
    joints = []
    for i in range(25):
        joint = [float(x) * scale for x in f.readline().split()]
        joints.append(glm.vec3(-joint[0], joint[1], joint[2]))

    line = f.readline()
    unposed_list.append(joints)

line = f.readline()    # Space
for j in range(frames):
    cam = [float(x) * scale for x in f.readline().split()]
    camera_list.append(glm.vec3(cam[0], cam[1], cam[2]))

f.close()


# Pre-processing for smooth motions
# Not sure this is necessary

# joints_Dance Specific Pre-processing
# awk_list = [115, 302, 306]
# interpolation_list = [[164, 170], [327, 332]]

# for awk in awk_list:
#     for i in range(len(posed_list[awk])):
#         posed_list[awk][i] = 0.5 * (posed_list[awk - 1][i] + posed_list[awk + 1][i])
#         unposed_list[awk][i] = 0.5 * (unposed_list[awk - 1][i] + unposed_list[awk + 1][i])

# for start, end in interpolation_list:
#     length = end - start
#     for j in range(start + 1, end):
#         for i in range(len(posed_list[awk])):
#             posed_list[j][i] = (end - j) / length * posed_list[start][i] + (j - start) / length * posed_list[end][i]
#             unposed_list[j][i] = (end - j) / length * unposed_list[start][i] + (j - start) / length * unposed_list[end][i]

# Apply simple filter for all lists
for j in range(frames):
    if j == 0:
        for i in range(25):
            posed_list[j][i] = 0.75 * posed_list[j][i] + 0.25 * posed_list[j + 1][i]
            unposed_list[j][i] = 0.75 * unposed_list[j][i] + 0.25 * unposed_list[j + 1][i]
        camera_list[j] = 0.75 * camera_list[j] + 0.25 * camera_list[j + 1]
    elif j == frames - 1:
        for i in range(25):
            posed_list[j][i] = 0.25 * posed_list[j - 1][i] + 0.75 * posed_list[j][i]
            unposed_list[j][i] = 0.25 * unposed_list[j - 1][i] + 0.75 * unposed_list[j][i]
        camera_list[j] = 0.25 * camera_list[j - 1] + 0.75 * camera_list[j]
    else:
        for i in range(25):
            posed_list[j][i] = 0.25 * posed_list[j - 1][i] + 0.5 * posed_list[j][i] + 0.25 * posed_list[j + 1][i]
            unposed_list[j][i] = 0.25 * unposed_list[j - 1][i] + 0.5 * unposed_list[j][i] + 0.25 * unposed_list[j + 1][i]
        camera_list[j] = 0.25 * camera_list[j - 1] + 0.5 * camera_list[j] + 0.25 * camera_list[j + 1]

# joints_Dance Specific Pre-processing
# camera_list[306] = 0.5 * (camera_list[305] + camera_list[307])


# Move posed and unposed list roots to origin (0, 0, 0) for efficiency
for j in range(frames):
    posed_copy = copy.deepcopy(posed_list)
    unposed_copy = copy.deepcopy(unposed_list)

    posed_offset = posed_copy[j][8]
    unposed_offset = unposed_copy[j][8]

    for i in range(25):
        posed_list[j][i] -= posed_offset
        unposed_list[j][i] -= unposed_offset


# This list saves all T pose joints (not rotated) aligned to XYZ axis
# [8, 1] edge aligned to Y axis, Facing Z axis
skeleton_list = copy.deepcopy(unposed_list)
skeleton_list = [skeleton_list[0]] * frames

# Unposed list is rotated in all xyz direction
# Align only [8, 1] edge to Y axis
for j in range(frames):
    vert = glm.normalize(unposed_list[j][1])
    unity = glm.vec3(0, 1, 0)

    cos_ = glm.dot(vert, unity)
    sin_ = np.sqrt(1 - cos_ * cos_)

    angle = m.atan2(sin_, cos_) # radians
    axis = glm.normalize(glm.cross(vert, unity))
    #angle = np.degrees(angle)

    rot = glm.mat4(1.0)
    rot = glm.rotate(rot, angle, axis)

    for i in range(25):
        unposed_list[j][i] = glm.vec3(rot * glm.vec4(unposed_list[j][i], 1.0))


# Aligning Skeleton list to XYZ axis
for j in range(frames):
    vert = glm.normalize(skeleton_list[j][1])
    unity = glm.vec3(0, 1, 0)

    cos_ = glm.dot(vert, unity)
    sin_ = np.sqrt(1 - cos_ * cos_)

    angle = m.atan2(sin_, cos_) # radians
    axis = glm.normalize(glm.cross(vert, unity))
    #angle = np.degrees(angle)

    rot = glm.mat4(1.0)
    rot = glm.rotate(rot, angle, axis)

    for i in range(25):
        skeleton_list[j][i] = glm.vec3(rot * glm.vec4(skeleton_list[j][i], 1.0))

for j in range(frames):
    neck = glm.normalize(glm.cross(skeleton_list[j][1], skeleton_list[j][0] - skeleton_list[j][1]))
    unitx = glm.vec3(1, 0, 0)

    cos_ = glm.dot(neck, unitx)
    sin_ = np.sqrt(1 - cos_ * cos_)

    angle = m.atan2(sin_, cos_) # radians
    axis = glm.normalize(glm.cross(neck, unitx))

    rot = glm.mat4(1.0)
    rot = glm.rotate(rot, angle, axis)

    for i in range(25):
        skeleton_list[j][i] = glm.vec3(rot * glm.vec4(skeleton_list[j][i], 1.0))



# Read output file from SMPLify-X
f = open('joints.bvh', 'w')

# For legibility
def tw(f, string, num):
    for i in range(num):
        f.write('	')
    f.write(string + '\n')

# Save joint information
class Joint():
    def __init__(self, num):
        self.num = num
        self.name = OPENPOSE[num]
        self.parent = None
        self.children = []
        self.depth = 0
        self.offset = glm.vec3(0)
        self.channels = []

    def print(self):
        print('num:', self.num)
        print('name:', self.name)
        if self.parent != None:
            print('parent:', self.parent.name)
        if len(self.children) > 0:
            print('children:', end=' ')
            for child in self.children:
                print(child.num, end=' ')
            print()
        print('depth:', self.depth)
        print('offset:', self.offset)
        print('channels:', self.channels)
        print()

# List of Joints
hierarchy = [0] * 25
hierarchy[8] = Joint(8)
# Only root has translation information
hierarchy[8].channels = 'Xposition Yposition Zposition Zrotation Yrotation Xrotation'.split()
frame_order = [8]

for edge in edges:
    num = edge[1]
    hierarchy[num] = Joint(num)
    hierarchy[num].parent = hierarchy[edge[0]]
    hierarchy[num].depth = hierarchy[num].parent.depth + 1

    for j in range(frames):
        hierarchy[num].offset += skeleton_list[j][num] - skeleton_list[j][edge[0]]
    hierarchy[num].offset /= frames
    #hierarchy[num].offset = skeleton_list[0][num] - skeleton_list[0][edge[0]]

    if edge[0] != 8:
        hierarchy[edge[0]].channels = 'Zrotation Yrotation Xrotation'.split()
    hierarchy[edge[0]].children.append(hierarchy[num])
    frame_order.append(num)

#print(frame_order)

# for joint in hierarchy:
#     joint.print()

# Case specific Joint information writing function
def jw(f, joint, num):
    if joint.depth == 0:
        tw(f, 'ROOT ' + joint.name, 0)
        tw(f, '{', 0)
        tw(f, 'OFFSET 0.0 0.0 0.0', 1)
        tw(f, 'CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation', 1)
        for child in joint.children:
            jw(f, child, num + 1)

    elif len(joint.parent.children) > 1:
        tw(f, 'JOINT ' + joint.parent.name + '_' + joint.name + '_joint', num)
        tw(f, '{', num)
        tw(f, 'OFFSET 0.0 0.0 0.0', num + 1)
        tw(f, 'CHANNELS 3 Zrotation Yrotation Xrotation', num + 1)

        offset = joint.offset

        if len(joint.children) == 0:
            tw(f, 'End Site', num + 1)
            tw(f, '{', num + 1)
            tw(f, 'OFFSET ' + str(np.round(offset[0], 6)) + ' ' + \
                            str(np.round(offset[1], 6)) + ' ' + \
                            str(np.round(offset[2], 6)), num + 2)
        else:
            tw(f, 'JOINT ' + joint.name, num + 1)
            tw(f, '{', num + 1)
            tw(f, 'OFFSET ' + str(np.round(offset[0], 6)) + ' ' + \
                            str(np.round(offset[1], 6)) + ' ' + \
                            str(np.round(offset[2], 6)), num + 2)
            tw(f, 'CHANNELS 3 Zrotation Yrotation Xrotation', num + 2)

            for child in joint.children:
                jw(f, child, num + 2)

        tw(f, '}', num + 1)

    elif len(joint.children) == 0:
        offset = joint.offset
        tw(f, 'End Site', num)
        tw(f, '{', num)
        tw(f, 'OFFSET ' + str(np.round(offset[0], 6)) + ' ' + \
                          str(np.round(offset[1], 6)) + ' ' + \
                          str(np.round(offset[2], 6)), num + 1)
    else:
        offset = joint.offset
        tw(f, 'JOINT '+ joint.name, num)
        tw(f, '{', num)
        tw(f, 'OFFSET ' + str(np.round(offset[0], 6)) + ' ' + \
                          str(np.round(offset[1], 6)) + ' ' + \
                          str(np.round(offset[2], 6)), num + 1)
        tw(f, 'CHANNELS 3 Zrotation Yrotation Xrotation', num + 1)
        for child in joint.children:
            jw(f, child, num + 1)

    tw(f, '}', num)

tw(f, 'HIERARCHY', 0)
jw(f, hierarchy[8], 0)
tw(f, 'MOTION', 0)
tw(f, 'Frames: ' + str(frames), 0)
tw(f, 'Frame Time: 0.033333', 0)

# Saves Root Y rotation angles for continuous rotation
t2_list = []
for j in range(frames):
    f.write('0 0 0 ')
    # f.write(str(np.round(-camera_list[j][0], 6)) + ' ' +\
    #         str(np.round(-camera_list[j][1], 6)) + ' ' +\
    #         str(np.round(-camera_list[j][2], 6)) + ' ')
    rot = glm.mat4(1.0)
    
    # For approximate model aligning,
    # decided to rotate skeleton to unposed joints
    # From there, joint aligning starts
    rot[0] = glm.vec4(glm.normalize(glm.cross(unposed_list[j][1], unposed_list[j][0] - unposed_list[j][1])), 0.0)
    rot[1] = glm.vec4(glm.normalize(unposed_list[j][1]), 0.0)
    rot[2] = glm.vec4(glm.normalize(glm.cross(glm.vec3(rot[0]), glm.vec3(rot[1]))), 0.0)

    rot0 = glm.mat4(1.0)
    if j != 0:
        rot0[0] = glm.vec4(glm.normalize(glm.cross(unposed_list[j - 1][1], unposed_list[j - 1][0] - unposed_list[j - 1][1])), 0.0)
        rot0[1] = glm.vec4(glm.normalize(unposed_list[j - 1][1]), 0.0)
        rot0[2] = glm.vec4(glm.normalize(glm.cross(glm.vec3(rot0[0]), glm.vec3(rot0[1]))), 0.0)

    rot = rot * glm.inverse(rot0)
    
    _, t2, _ = rotation_zyx(rot)

    # For continuous rotation angle (no clamping at pi or -pi)
    # difference from previous frame is added to Y rotation angle for previous frame
    # This enables rotation angles like -400... joints_Dance is a good example
    if j != 0:
        t2 += t2_list[-1]

    t2_list.append(t2)
    f.write('0 ' + str(np.round(t2, 6)) + ' 0 ')

    rot = glm.rotate(glm.mat4(1.0), np.radians(t2), glm.vec3(0, 1, 0))
    
    # Saves change of coordinates for parent joint
    Matrices = [rot] * 25
    for edge in edges:
        s = edge[0]
        e = edge[1]

        before = glm.normalize(hierarchy[e].offset)
        after = glm.normalize(posed_list[j][e] - posed_list[j][s])

        # I haven't considered change of coordinates...
        after = glm.normalize(glm.inverse(Matrices[s]) * after)

        cos_ = glm.dot(before, after)
        sin_ = np.sqrt(1 - cos_ * cos_)
        angle = np.arctan2(sin_, cos_) # radians

        axis = glm.normalize(glm.cross(before, after))
        #angle = np.degrees(angle)

        rot_ = glm.rotate(glm.mat4(1.0), angle, axis)
        t1, t2, t3 = rotation_zyx(rot_)
        
        # BVH ZYX rotation should be thought locally
        # That is, rotate for Z axis first
        # Next, rotate for "Local" Y axis that is already rotated in the first step!
        # Last, rotate for X axis
        # When reading Matrices, this should be written from left to right
        # Therefore, rotation matrix is *** RzRyRx ***
        # Rotation angle should be also calculated from this formula

        # Drawing BVH file also works in this order!
        # Rotate z, rotate y, rotate x, draw line from origin to offset
        Matrices[e] =  Matrices[s] * rot_

        f.write(str(np.round(t1, 6)) + ' ' +\
                str(np.round(t2, 6)) + ' ' +\
                str(np.round(t3, 6)) + ' ') 
                
        if len(hierarchy[e].children) > 1:
            f.write('0 0 0 ')

    f.write('\n')

f.close()


# Visualization part
def reshape(w, h):
    global width
    global height
    width = glutGet(GLUT_WINDOW_WIDTH)
    height = glutGet(GLUT_WINDOW_HEIGHT)

    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(viewangle, w / h, 1.0, 100000.0)
    gluLookAt(view.x * d, view.y * d, view.z * d, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


def init():
    glClearColor(1.0, 1.0, 1.0, 1.0)
    glShadeModel(GL_SMOOTH)
    glEnable(GL_COLOR_MATERIAL)
    glEnable(GL_DEPTH_TEST)

    glutPostRedisplay()

def display(): 
    global nowframe
    global viewangle
    global width, height
    global view, d, lookat, fix_d, rot_angle
    global posed_list, unposed_list
    global camera_list
    global edges
    glClearColor(1.0, 1.0, 1.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)


    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective (viewangle,  width/height, 1.0, 100000.0)
    if camera_fix:   
        # gluLookAt(0, 0, fix_d , lookat.x, lookat.y, lookat.z, 0.0, 1.0, 0.0)
        gluLookAt(fix_d * np.sin(rot_angle), 0, fix_d * np.cos(rot_angle), lookat.x, lookat.y, lookat.z, 0.0, 1.0, 0.0)
        # view = camera_list[0]
        # gluLookAt(view[0] * d, view[1] * d, view[2] * d, lookat.x, lookat.y, lookat.z, 0.0, 1.0, 0.0)
    else:
        view = camera_list[nowframe]
        gluLookAt(view[0] * d, view[1] * d, view[2] * d, lookat.x, lookat.y, lookat.z, 0.0, 1.0, 0.0)
    #gluLookAt(-view[0] * d, -view[1] * d, -view[2] * d, lookat.x, lookat.y, lookat.z, 0.0, -1.0, 0.0)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    glPointSize(5)
    glColor3f(0, 0, 0)
    glBegin(GL_POINTS)

    joint_list = None
    if posed:
        joint_list = posed_list
        #joint_list = unposed_list
    else:
        joint_list = unposed_list
        #joint_list = skeleton_list
    for joint in joint_list[nowframe]:
        glVertex3f(joint[0], joint[1], joint[2])

    glEnd()

    glColor3f(0.0, 0.0, 1.0)
    glBegin(GL_LINES)
    for edge in edges:
        s = joint_list[nowframe][edge[0]]
        e = joint_list[nowframe][edge[1]]
        glVertex3f(s[0], s[1], s[2])
        glVertex3f(e[0], e[1], e[2])
    glEnd()

    glBegin(GL_LINES)
    glColor3f(1,0,0)
    glVertex3f(0,0,0)
    glVertex3f(100,0,0)

    glColor3f(0,1,0)
    glVertex3f(0,0,0)
    glVertex3f(0,100,0)

    glColor3f(0,0,1)
    glVertex3f(0,0,0)
    glVertex3f(0,0,100)

    # For Checking alignment of skeleton_list and unposed_list
    # Unnecessary for result

    # glColor3f(0,0,0)
    # front = glm.normalize(glm.cross(unposed_list[nowframe][1], unposed_list[nowframe][0] - unposed_list[nowframe][1]))
    # front = 100 * glm.normalize(glm.cross(front, unposed_list[nowframe][1]))
    # glVertex3f(0,0,0)
    # glVertex3f(front[0], front[1], front[2])

    # glColor3f(0.5,0.5,0.5)
    # front = glm.normalize(glm.cross(skeleton_list[nowframe][1], skeleton_list[nowframe][0] - skeleton_list[nowframe][1]))
    # front = 100 * glm.normalize(glm.cross(front, skeleton_list[nowframe][1]))
    # glVertex3f(0,0,0)
    # glVertex3f(front[0], front[1], front[2])

    glEnd()

    glutSwapBuffers()

def keyboard(key, x, y):
    if key == b' ':
        global framenum
        framenum = 1 - framenum
    elif key == b'j':
        global nowframe
        for joints in posed_list[nowframe]:
            print(joints)
        print('camera:', camera_list[nowframe])
    elif key == b'p':
        global posed
        posed = not posed
    elif key == b'c':
        global camera_fix
        camera_fix = not camera_fix
    elif key == b'r':
        nowframe = 0

    elif key == b'w':
        global rot_angle
        rot_angle= -np.pi
    elif key == b's':
        rot_angle = 0
    elif key == b'a':
        rot_angle -= np.pi / 4
    elif key == b'd':
        rot_angle += np.pi / 4

def special(key, x, y):
    global framenum
    if framenum == 0:
        global nowframe
        if key == 100:
            nowframe -= 1
            if nowframe < 0:
                nowframe += frames

            print(nowframe)
        elif key == 102:
            nowframe += 1
            if nowframe >= frames:
                nowframe -= frames

            print(nowframe)
            

def timer(value):
    global nowframe, framenum
    global camera_fix, rot_angle
    nowframe += framenum
    if nowframe >= frames:
        nowframe -= frames
    if nowframe < 0:
        nowframe += frames

    glutPostRedisplay()
    glutTimerFunc(int(frametime * 1000), timer, 0)

if __name__ == "__main__":
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
    glutInitWindowSize(width, height)
    glutInitWindowPosition(150, 150)
    glutCreateWindow("Joint Viewer")
    init()
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glutSpecialFunc(special)
    glutTimerFunc(0, timer, 0)
    glutMainLoop()