# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 13:21:30 2022

@author: mario
"""
import vtk
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtk.util import numpy_support
import numpy as np
import math
import random
import os


################################################################################
################################################################################
################################################################################
#    GLOBAL VARIABLES

# Wheter to save images to disk or just show them on screen
save_to_disk = True


# Skull dataset path
dataset_path = "C:/Users/mario/Documents/mario/ingenieria_informatica/TFG/dataset/PM_DATA/"

# Image generation path
image_path = "C:/Users/mario/Documents/mario/ingenieria_informatica/TFG/dataset/PM_DATA_IMAGES/"

# Dataset information path
info_path = "C:/Users/mario/Documents/mario/ingenieria_informatica/TFG/dataset/informacion_dataset.xlsx"

# Number of available models
num_models = 117

# List with posibles bad models not to be considered
bad_models = [24, 40, 106]


# Window size (also image size when saved to disk), being a list with the width and height
window_size = [224, 224]

# Window background color 
ColorBackground = vtkNamedColors().GetColor3d('Black')


# Number of images generated per cranium 3D model
#   Half of them are generated with the camera at 0.6 meters of the model and
#   the other half with the camera at 3 meters of the model
num_images = 100


# Maximum rotation angle (horizontal and vertical) of the camera when generating sequence of images
rotation_angle = 10


# Distances of the camera
cam_distances = [600, 3000]




################################################################################
################################################################################
################################################################################
#    FUNCTION DECLARATION


def get_polydata_trans(actor):
    # Create poly data and copy actor's polydata
    polyData = vtk.vtkPolyData()
    polyData.DeepCopy(actor.GetMapper().GetInput())
    
    # Get actors transformation
    transform = vtk.vtkTransform()
    transform.SetMatrix(actor.GetMatrix())
    
    # Apply transformation to poly data
    fil = vtk.vtkTransformPolyDataFilter()
    fil.SetTransform(transform)
    fil.SetInputDataObject(polyData)
    fil.Update()
    
    # Copy transformed poly data
    polyData.DeepCopy(fil.GetOutput())
    
    return polyData


################################################################################
# Get the coordinates of each of the bound points in a 3D model.
# Parameters:
#   data, VTK's PolyData object
# Returns:
#   bound_x_min, coordinates of minimun x point
#   bound_x_max, coordinates of maximum x point
#   bound_y_min, coordinates of minimun y point
#   bound_y_max, coordinates of maximum y point
#   bound_z_min, coordinates of minimun z point
#   bound_z_max, coordinates of maximum z point
def get_bounds_coords(data):
    # Obtain all data points
    points = numpy_support.vtk_to_numpy(data.GetPoints().GetData())
    
    # Get points with minimum x coordinate
    bound_x_min = points[np.argmin(points[:,0])]
    # Get points with maximum x coordinate
    bound_x_max = points[np.argmax(points[:,0])]
    
    # Get points with minimum y coordinate
    bound_y_min = points[np.argmin(points[:,1])]
    # Get points with maximum y coordinate
    bound_y_max = points[np.argmax(points[:,1])]
    
    # Get points with minimum z coordinate
    bound_z_min = points[np.argmin(points[:,2])]
    # Get points with maximum z coordinate
    bound_z_max = points[np.argmax(points[:,2])]
    
    return bound_x_min, bound_x_max, bound_y_min, bound_y_max, bound_z_min, bound_z_max

    

################################################################################
# Obtain the coordinates of the 3D object center. It is calculated 
# as the middle point between the X,Y,Z bounds of the 3D model.
# Parameters:
#   data, VTK's PolyData object
# Returns:
#   center, the center of the 3D object
def get_center(data):
    # Get model bounds in each axis
    bounds = [0] * 6
    data.GetCellsBounds(bounds)
    
    center = ( (bounds[1] + bounds[0]) / 2,
               (bounds[3] + bounds[2]) / 2,
               (bounds[5] + bounds[4]) / 2 )
    
    return center


################################################################################
# Check if 3D model is entirely within the visualization window.
# Parameters:
#   data, VTK's PolyData object
#   view_port, view port of the scene
#   window_dim, dimensions of the visualization window
# Returns:
#   is_model_inside, whether the 3D model is inside the window or not (boolean)
def check_model_inside_window(data, view_port, window_dim):
    # Get model bounds in each axis
    bounds_coord = get_bounds_coords(data)
    
    # Initialize vtkCoordinate and set it to get World Coordinates
    vtkCoord = vtk.vtkCoordinate()
    vtkCoord.SetCoordinateSystemToWorld()
    
    is_model_inside = True
    
    # For each bound of the model
    for bound in bounds_coord:
        vtkCoord.SetValue(bound)
        coordWorld = vtkCoord.GetComputedDisplayValue(view_port)
        
        is_model_inside = is_model_inside and coordWorld[0] < window_dim[0] and \
                                              coordWorld[1] < window_dim[1]
        
    return is_model_inside 



################################################################################
# Obtain minimum distance at which camera should be positioned in order
# to visualize the whole model, taking into account that the camera is
# positioned parallel to Z axis in the 3D object center.
# For each point of the model the distance is calculated and the maximum
# of these distances is returned so each point is visible. The distance
# calculated for each point should consider the X and Y position of the
# point separately as the following
#   i. The corresponing component (x or y) of the center of the model
#           is substracted from the respective component of the point
#           and the absolute value is taken.
#   ii. Divide it by the tangent of half the camera vision angle,
#           which will return the relative distance of the camera to
#           the point.
#   iii. Add the Z component of the point to obtain the absolute 
#           distance of the camera to XY plane.
# Parameters:
#   data, VTK's PolyData object
# Returns:
#   dist, the position of the camera at which the distance
#           to the model is minimum 
def get_min_pos(data, angle):
    # Obtain all data points
    points = numpy_support.vtk_to_numpy(data.GetPoints().GetData())
    
    # Get center of the points
    center_point = get_center(data)
    
    # Calculate tangent of half the camera view angle
    radiants = (angle/2 * math.pi) / 180
    tangent = math.tan(radiants)
    
    # Get minimum distance for each point considering x coordinate
    distances_x = ( abs(points[:,0]-center_point[0]) / tangent) + points[:,2] 
    # Get minimum distance for each point considering y coordinate
    distances_y = ( abs(points[:,1]-center_point[1]) / tangent) + points[:,2] 
    
    # Obtain maximum distance of all the minimum distances
    dist = max(np.max(distances_x), np.max(distances_y))
    
    # 1 pixel is added to distance in order to separete 1 pixel the model from the window border
    return dist + 1



################################################################################
# Obtain minimum view angle at which 3D model fills 90% the visualization
# window for a specific camera position.
# For each point of the model the angle is calculated and the maximum
# of these angles is returned so each point is visible. The angle
# calculated for each point should consider the X and Y position of that
# point separately as the following
#   i. Get opposite cathetus of the angle as difference of the point 
#           projection and the camera projection in z-perpendicular plane.
#   ii. Get adjacent cathetus of the angle as the difference between
#           z-coordinate of camera and point.
#   iii. Get view angle as the arctangent after dividing the opposite 
#           cathetus by the adjacent one.
# Parameters:
#   data, VTK's PolyData object
#   cam_pos, camera position
# Returns:
#   view_angle, minimum view angle of the camera to fit model in window
def get_min_angle_view(data, cam_pos):
    # Obtain all data points
    points = numpy_support.vtk_to_numpy(data.GetPoints().GetData())
    
    # # Get minimum angle for each point considering x coordinate
    # angles_x = np.arctan(abs(points[:,0]-cam_pos[0]) / (0.9 * cam_pos[2] - points[:,2])) # Multiplied by 0.9 to fit 90%
    # # Get minimum angle for each point considering y coordinate
    # angles_y = np.arctan(abs(points[:,1]-cam_pos[1]) / (0.9 * cam_pos[2] - points[:,2])) # Multiplied by 0.9 to fit 90%
    
    # Get minimum angle for each point considering x coordinate
    angles_x = np.arctan(abs(points[:,0]-cam_pos[0]) / (0.9 * abs(cam_pos[2] - points[:,2]))) # Multiplied by 0.9 to fit 90%
    # Get minimum angle for each point considering y coordinate
    angles_y = np.arctan(abs(points[:,1]-cam_pos[1]) / (0.9 * abs(cam_pos[2] - points[:,2]))) # Multiplied by 0.9 to fit 90%
    
    # Obtain maximum of all the minimum angles
    angle = max(np.max(angles_x), np.max(angles_y))
    
    # Convert to degrees and double its value to get the view angle
    view_angle = (angle * 180 / math.pi) * 2
    
    return view_angle 



################################################################################
# Calculate maximum displacement in X axis and Y axis.
# The function obtains the max distance for which model can be translated in 
# X-axis and Y-axis in each direction while remaining inside the window of
# visualization. The function calculates four distances: -x offset, x offset,
# -y offset and y offset. For each point of the model this four distances are
# calculated individually and the minimum of each distance is kept.
# Parameters:
#   data, VTK's PolyData object
#   camera_pos, position of the camera in world coordinates
#   angle, angle of view of the camera
# Returns:
#   max_offset_neg_x, maximum -x offset (negative displacement)
#   max_offset_pos_x, maximum  x offset (positive displacement)
#   max_offset_neg_y, maximum -y offset (negative displacement)
#   max_offset_pos_y, maximum  y offset (positive displacement)
def get_max_desp(data, camera_pos, angle):
    # Obtain all data points
    points = numpy_support.vtk_to_numpy(data.GetPoints().GetData())
    
    # Get half the view angle in radiants
    radiants = (angle/2 * math.pi) / 180
    tangent = math.tan(radiants)
    
    
    # Get points which are left of the camera
    points_left  = points[np.where(points[:,0] < camera_pos[0])]
    # Obtain x offsets for these points
    offsets_neg_x = - abs(- abs(camera_pos[2] - points_left[:,2]) * tangent - (points_left[:,0] - camera_pos[0]))
    # Get minimum offset (in prctice the maximum as they are negative distances)
    max_offset_neg_x = np.max(offsets_neg_x)
    
    # Get points which are right of the camera
    points_right = points[np.where(points[:,0] > camera_pos[0])]
    # Obtain x offsets for these points
    offsets_pos_x = abs(abs(camera_pos[2] - points_right[:,2]) * tangent - (points_right[:,0] - camera_pos[0]))
    # Get minimum offset
    max_offset_pos_x = np.min(offsets_pos_x)
    
    # Get points which are under the camera
    points_under = points[np.where(points[:,1] < camera_pos[1])]
    # Obtain x offsets for these points
    offsets_neg_y = - abs(- abs(camera_pos[2] - points_under[:,2]) * tangent - (points_under[:,1] - camera_pos[1]))
    # Get minimum offset (in prctice the maximum as they are negative distances)
    max_offset_neg_y = np.max(offsets_neg_y)
    
    # Get points which are above the camera
    points_above = points[np.where(points[:,1] > camera_pos[1])]
    # Obtain x offsets for these points
    offsets_pos_y = abs(abs(camera_pos[2] - points_above[:,2]) * tangent - (points_above[:,1] - camera_pos[1]))
    # Get minimum offset
    max_offset_pos_y = np.min(offsets_pos_y)
    
    
    return max_offset_neg_x, max_offset_pos_x, max_offset_neg_y, max_offset_pos_y


################################################################################
################################################################################
################################################################################
#    SCRIPT


# Check if image directory is already created and if not create it
if not os.path.isdir(image_path):
    os.mkdir(image_path)


# Iterate over each cranium model
for n_skull in range(97, num_models+1):
    # If the actual model is bad the iteration is skipped
    if n_skull in bad_models:
        continue
    
    print("Starting with " + str(n_skull) + "th cranium...")
    
    # Get path where skull model is saved in disk
    skull_path = dataset_path + "+PM" + str(n_skull) + "/skull/"
    
    # Get absolute path of posible aligned skull file
    fname_aligned = skull_path + "skull_" + str(n_skull) + "_aligned.obj"
    
    # Check if exists aligned skull file
    if os.path.isfile(fname_aligned):
        fname_skull = fname_aligned
    else: # Else get the normal skull file absolute path
        fname_skull = skull_path + "skull_" + str(n_skull) + ".obj"
        
    
    ################################################################################
    # Read 3D model (cranium)
    reader = vtk.vtkOBJReader()
    
    reader.SetFileName(fname_skull)

    reader.Update()
    
    
    ################################################################################
    # Create a mapper from read data
    mapper = vtk.vtkPolyDataMapper()
    
    mapper.SetInputConnection(reader.GetOutputPort()) # Allocate reader output
    
    
    ################################################################################
    # Get center of the 3D model
    center = get_center(mapper.GetInput())
    
    
    ################################################################################
    # Create actor to show model in the window
    actor_cranium = vtk.vtkActor()
    
    actor_cranium.SetMapper(mapper) # Allocate mapper
    
    
    ################################################################################
    # Create visualization window
    window = vtk.vtkRenderWindow()
    
    window.SetSize(window_size[0],window_size[1]) # Adjust window size
    
    # Create Interactor to capture defaults mouse events
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)
    
    # Create renderer
    renderer = vtk.vtkRenderer()
    window.AddRenderer(renderer) # Add it to window
    renderer.AddActor(actor_cranium) # Add actor to renederer
    renderer.SetBackground(ColorBackground) # Adjust background color to window
    renderer.ResetCamera()
    
    # Set camera focal point to the center of the model
    renderer.GetActiveCamera().SetFocalPoint(center)
    
    # Create window
    window.Render()
    
    
    
    ################################################################################
    ################################################################################
    # Generate images out of the model by applying random transformations and 
    #   changing the camera position
    
    # Get individual name
    ind_name = 'PM' + str(n_skull)
    
    images_gen = 0
    
    while images_gen < num_images:
        # One image for each distance
        for dist in cam_distances:
            image_taken = False # To check if the model is inside visualization window (if not image is discarded)
        
            while not image_taken:
                # Generate random rotation angles in range
                x_angle = random.SystemRandom().uniform(-rotation_angle, rotation_angle)
                y_angle = random.SystemRandom().uniform(-rotation_angle, rotation_angle)
                z_angle = random.SystemRandom().uniform(-rotation_angle, rotation_angle)
                
                # Rotate camera horizontally and vertically
                actor_cranium.RotateX(x_angle)
                actor_cranium.RotateY(y_angle)
                actor_cranium.RotateZ(z_angle)
                
                # Get cranium bounds in each axis
                bounds = actor_cranium.GetBounds()
                
                # Set camera at dist of the model
                camera_pos = list(center) # X and Y coordinates equal as center of model
                camera_pos[2] = bounds[5] + dist # Z coordinate separated dist from Z positive bound of model
                
                renderer.GetActiveCamera().SetPosition(camera_pos)
                
                # Modify view angle for model to fit the window
                new_view_angle = get_min_angle_view(get_polydata_trans(actor_cranium), camera_pos)
                renderer.GetActiveCamera().SetViewAngle(new_view_angle)
                
                # Get maximum displacements of cranium
                max_desp_neg_x, max_desp_pos_x, \
                    max_desp_neg_y, max_desp_pos_y = get_max_desp(get_polydata_trans(actor_cranium),
                                                                  camera_pos,
                                                                  new_view_angle)
                
                # Generate random translations
                desp_x = random.SystemRandom().uniform(max_desp_neg_x, max_desp_pos_x)
                desp_y = random.SystemRandom().uniform(max_desp_neg_y, max_desp_pos_y)
                
                # Translate the model
                actor_cranium.AddPosition(desp_x, desp_y, 0)
                
                # Modify camera clipping planes to visualize data as close as 1 pixel away from the camera
                # and visualize the farthest point of the model
                renderer.GetActiveCamera().SetClippingRange( (1, abs(camera_pos[2]-bounds[4])+1) )
                
                # Check if whole model is inside the visualization window 
                image_taken = check_model_inside_window(get_polydata_trans(actor_cranium), renderer, window_size)
                
                # If model is inside window, image is saved
                if image_taken and save_to_disk:
                    # Check if image directory exists and if not create it
                    image_dir = image_path + "PM" + str(n_skull)
                    
                    if not os.path.isdir(image_dir):
                        os.mkdir(image_dir)
                    
                    # Save image to disk
                    w2if = vtk.vtkWindowToImageFilter()
                    w2if.SetInput(window)
                    w2if.Update()
                    
                    writer = vtk.vtkPNGWriter()
                    writer.SetFileName(image_dir + "/skull_" + str(n_skull) + "_image_" + str(int(images_gen)) + ".png")
                    writer.SetInputData(w2if.GetOutput())
                    writer.Write()
                
                # Undo rotations
                actor_cranium.RotateX(-x_angle)
                actor_cranium.RotateY(-y_angle)
                actor_cranium.RotateZ(-z_angle)
                
                # Undo translation
                actor_cranium.AddPosition(-desp_x, -desp_y, 0)
                
            # One image has being generated
            images_gen += 1
    
    
    window.Finalize()  
    interactor.TerminateApp()
    del window, interactor
    
    print("Finished with " + str(n_skull) + "th cranium.")
   
    
   
# ################################################################################
# # Start the event listening of the Interactor
# interactor.Start()