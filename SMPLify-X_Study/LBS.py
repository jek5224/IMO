from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import glm
import numpy as np
import copy
import time

width = 1500
height = 800

scale = 100

frames = 10
frametime = 1 / 30

nowframe = 0
framenum = 1

view = glm.normalize(glm.vec3(0.0, 0.0, 1.0))
lookat = glm.vec3(0.0, 0.0, 0.0)
viewangle = 60.0
d = 30

LBS = False
quad = False

up_joint = glm.vec3(-20, 0, 0)
down_joint = glm.vec3(0, 0, 0)
up_vertices = []
down_vertices = []

up_rotation = 0
down_rotation = 0

vertex_n = 100

# Defining in local coordinate
# for i in range(vertex_n + 1):
#     up_vertices.append(glm.vec3(20 - 20 * i / vertex_n, 5, 0))
# for i in range(vertex_n + 1):
#     up_vertices.append(glm.vec3(20 - 20 * i / vertex_n, -5, 0))

# Global coordinate of vertices
slice_n = 30
for k in range(slice_n):
    for i in range(vertex_n):
        up_vertices.append(up_joint + glm.vec3(20 * i / vertex_n, \
                                               5 * np.cos(k / slice_n * 2 * np.pi), \
                                               5 * np.sin(k / slice_n * 2 * np.pi)))

for k in range(slice_n):
    for i in range(vertex_n):
        down_vertices.append(down_joint + glm.vec3(20 * i / vertex_n, \
                                               5 * np.cos(k / slice_n * 2 * np.pi), \
                                               5 * np.sin(k / slice_n * 2 * np.pi)))

w_up_no = []
w_down_no = []

for i in range(len(up_vertices)):
    w_up_no.append([1, 0])
for i in range(len(down_vertices)):
    w_down_no.append([0, 1])

w_up_LBS = copy.deepcopy(w_up_no)
w_down_LBS = copy.deepcopy(w_down_no)

w_up_LBS_quad = copy.deepcopy(w_up_LBS)
w_down_LBS_quad = copy.deepcopy(w_down_LBS)

up_curve_n = int(vertex_n * 1)
down_curve_n = int(vertex_n * 1)

for k in range(slice_n):
    for i in range(up_curve_n):
        w_up_LBS[(k + 1) * vertex_n - 1 - i][0] = 0.5 + 0.5 * i / up_curve_n
        w_up_LBS[(k + 1) * vertex_n - 1 - i][1] = 1 - w_up_LBS[(k + 1) * vertex_n - 1 - i][0]

        w_up_LBS_quad[(k + 1) * vertex_n - 1 - i][0] = -0.5 * (i / up_curve_n - 1) ** 2 + 1
        w_up_LBS_quad[(k + 1) * vertex_n - 1 - i][1] = 1 - w_up_LBS_quad[(k + 1) * vertex_n - 1 - i][0]

for k in range(slice_n):
    for i in range(down_curve_n):
        w_down_LBS[k * vertex_n + i][1] = 0.5 + 0.5 * i / up_curve_n
        w_down_LBS[k * vertex_n + i][0] = 1 - w_down_LBS[k * vertex_n + i][1]

        w_down_LBS_quad[k * vertex_n + i][1] = -0.5 * (i / down_curve_n - 1) ** 2 + 1
        w_down_LBS_quad[k * vertex_n + i][0] = 1 - w_down_LBS_quad[k * vertex_n + i][1]


def drawSquare(midx, midy, h, color, angle, axis):
    glPushMatrix()
    glColor3f(0,0,0)
    glRotatef(angle, axis[0], axis[1], axis[2])

    glBegin(GL_LINE_LOOP)
    glVertex3f(midx + h / 2, midy + h / 2, 0)
    glVertex3f(midx - h / 2, midy + h / 2, 0)
    glVertex3f(midx - h / 2, midy - h / 2, 0)
    glVertex3f(midx + h / 2, midy - h / 2, 0)
    glEnd()

    glColor3f(color[0], color[1], color[2])
    glBegin(GL_POLYGON)
    glVertex3f(midx + h / 2, midy + h / 2, 0)
    glVertex3f(midx - h / 2, midy + h / 2, 0)
    glVertex3f(midx - h / 2, midy - h / 2, 0)
    glVertex3f(midx + h / 2, midy - h / 2, 0)
    glEnd()
    glPopMatrix()

def drawCube(midx, midy, midz, h, color, angle, axis):
    glPushMatrix()
    glTranslatef(midx, midy, midz)
    glRotatef(angle, axis[0], axis[1], axis[2])
    glColor3f(color[0], color[1], color[2])
    glScalef(h, h, h)
    glutSolidCube(1.0)
    glColor3f(0,0,0)
    glutWireCube(1.0001)
    glPopMatrix()
    
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
    glClearColor(1.0, 1.0, 1.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective (viewangle,  width/height, 1.0, 100000.0)
    gluLookAt(view[0] * d, view[1] * d, view[2] * d, lookat.x, lookat.y, lookat.z, 0.0, 1.0, 0.0)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # For visualizing
    # True LBS should be performed in global coordinate!
    # Let's avoid using Matrix overlapping
    # glPointSize(5.0)

    # glPushMatrix()
    # glColor3f(0,0,1)
    # glTranslatef(up_joint[0], up_joint[1], up_joint[2])
    # glRotatef(up_rotation, 0,0,1)
    # glBegin(GL_LINES)
    # for i in range(1, vertex_n + 1):
    #     glVertex3f(up_vertices[i][0], up_vertices[i][1], up_vertices[i][2])
    #     glVertex3f(up_vertices[i - 1][0], up_vertices[i - 1][1], up_vertices[i - 1][2])

    #     glVertex3f(up_vertices[i + 1 + vertex_n][0], up_vertices[i + 1 + vertex_n][1], up_vertices[i + 1 + vertex_n][2])
    #     glVertex3f(up_vertices[i + vertex_n][0], up_vertices[i + vertex_n][1], up_vertices[i + vertex_n][2])
    # glEnd()

    # drawSquare(0,0,0.5, [0,0,1])

    # glPushMatrix()
    # glColor3f(1,0,0)
    # glTranslatef(down_joint[0] - up_joint[0], down_joint[1] - up_joint[1], down_joint[2] - up_joint[2])
    # glRotatef(down_rotation, 0,0,1)
    # glBegin(GL_LINES)
    # for i in range(1, vertex_n + 1):
    #     glVertex3f(down_vertices[i][0], down_vertices[i][1], down_vertices[i][2])
    #     glVertex3f(down_vertices[i - 1][0], down_vertices[i - 1][1], down_vertices[i - 1][2])

    #     glVertex3f(down_vertices[i + 1 + vertex_n][0], down_vertices[i + 1 + vertex_n][1], down_vertices[i + 1 + vertex_n][2])
    #     glVertex3f(down_vertices[i + vertex_n][0], down_vertices[i + vertex_n][1], down_vertices[i + vertex_n][2])
    # glEnd()

    # drawSquare(0,0,0.5, [1,0,0])

    # glPopMatrix() 

    # glPopMatrix()

    global w_up_no, w_down_no

    w_up = w_up_no
    w_down = w_down_no
    if LBS:
        if quad:
            global w_up_LBS_quad, w_down_LBS_quad
            w_up = w_up_LBS_quad
            w_down = w_down_LBS_quad
        else:
            global w_up_LBS, w_down_LBS
            w_up = w_up_LBS
            w_down = w_down_LBS

    #drawSquare(up_joint[0], up_joint[1], 0.5, glm.vec3(0,0,1), 0, glm.vec3(1,0,0))
    

    nowtime = time.time()
    angle = 70 + 70 * np.cos(nowtime)
    axis = glm.vec3(0, 0, 1)

    # angle = 20 * np.cos(nowtime)
    # axis = glm.normalize(glm.vec3(1, 0, 0))

    rot_mat = glm.rotate(glm.mat4(1.0), glm.radians(angle), axis)

    #drawSquare(down_joint[0], down_joint[1], 0.5, glm.vec3(1,0,0), angle, axis)
    drawCube(up_joint[0], up_joint[1], up_joint[2], 0.5, glm.vec3(0,0,1), 0, axis)
    drawCube(down_joint[0], down_joint[1], down_joint[2], 0.5, glm.vec3(1,0,0), angle, axis)

    glBegin(GL_LINES)
    glColor3f(0,0,1)

    up_copy = copy.deepcopy(up_vertices)

    for i in range(len(up_copy)):
        up_copy[i] = w_up[i][0] * up_vertices[i] + w_up[i][1] * rot_mat * up_vertices[i]

    for k in range(slice_n):
        for i in range(1, vertex_n):
            glVertex3f(up_copy[i + k * vertex_n][0],\
                    up_copy[i + k * vertex_n][1],\
                    up_copy[i + k * vertex_n][2])
            glVertex3f(up_copy[i - 1 + k * vertex_n][0], \
                    up_copy[i - 1 + k * vertex_n][1], \
                    up_copy[i - 1 + k * vertex_n][2])
            
    for i in range(vertex_n):
        for k in range(slice_n):
            if k != slice_n - 1:
                glVertex3f(up_copy[i + k * vertex_n][0],\
                           up_copy[i + k * vertex_n][1],\
                           up_copy[i + k * vertex_n][2])
                glVertex3f(up_copy[i + (k + 1) * vertex_n][0],\
                           up_copy[i + (k + 1) * vertex_n][1],\
                           up_copy[i + (k + 1) * vertex_n][2])
            else:
                glVertex3f(up_copy[i + k * vertex_n][0],\
                           up_copy[i + k * vertex_n][1],\
                           up_copy[i + k * vertex_n][2])
                glVertex3f(up_copy[i][0],\
                           up_copy[i][1],\
                           up_copy[i][2])

    down_copy = copy.deepcopy(down_vertices)
    for i in range(len(down_copy)):
        down_copy[i] = w_down[i][0] * down_vertices[i] + w_down[i][1] * rot_mat * down_vertices[i]

    glColor3f(0.8,0,0.8)
    for k in range(slice_n):
        glVertex3f(up_copy[(k + 1) * vertex_n - 1][0],\
                   up_copy[(k + 1) * vertex_n - 1][1],\
                   up_copy[(k + 1) * vertex_n - 1][2])
        glVertex3f(down_copy[k * (vertex_n)][0],\
                   down_copy[k * (vertex_n)][1],\
                   down_copy[k * (vertex_n)][2])
        
    glColor3f(1,0,0)

    for k in range(slice_n):
        for i in range(1, vertex_n):
            glVertex3f(down_copy[i + k * vertex_n][0],\
                    down_copy[i + k * vertex_n][1],\
                    down_copy[i + k * vertex_n][2])
            glVertex3f(down_copy[i - 1 + k * vertex_n][0], \
                    down_copy[i - 1 + k * vertex_n][1], \
                    down_copy[i - 1 + k * vertex_n][2])
            
    for i in range(vertex_n):
        for k in range(slice_n):
            if k != slice_n - 1:
                glVertex3f(down_copy[i + k * vertex_n][0],\
                           down_copy[i + k * vertex_n][1],\
                           down_copy[i + k * vertex_n][2])
                glVertex3f(down_copy[i + (k + 1) * vertex_n][0],\
                           down_copy[i + (k + 1) * vertex_n][1],\
                           down_copy[i + (k + 1) * vertex_n][2])
            else:
                glVertex3f(down_copy[i + k * vertex_n][0],\
                           down_copy[i + k * vertex_n][1],\
                           down_copy[i + k * vertex_n][2])
                glVertex3f(down_copy[i][0],\
                           down_copy[i][1],\
                           down_copy[i][2])
                
    glEnd()

    glBegin(GL_LINES)
    glColor3f(0,0,0)
    glVertex3f(0,0,0)
    glVertex3f(100,0,0)

    glVertex3f(0,0,0)
    glVertex3f(0,100,0)

    glVertex3f(0,0,0)
    glVertex3f(0,0,100)
    glEnd()

    glutSwapBuffers()

def keyboard(key, x, y):
    global LBS
    if key == b'l':
        LBS = not LBS
    elif LBS and key == b'q':
        global quad
        quad = not quad

def special(key, x, y):
    global framenum
    if framenum == 0:
        global nowframe
            

def timer(value):
    glutPostRedisplay()
    glutTimerFunc(int(1 / 30 * 1000), timer, 0)

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