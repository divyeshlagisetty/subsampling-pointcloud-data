from email import header
from tracemalloc import start
import numpy as np
import open3d as o3d
# from pandas import array
# import torch
import os
import time
# import octree_handler (added to the function which needs it)
import random
import pickle
# import dialogs as dia (added to the functions that need it)
from tqdm import tqdm 
# from plyfile import PlyData, PlyElement # Added to the function that needs it

cwd = os.getcwd() + "/" # Get current working directory
temp_loc = cwd + "/.temp"

def path(text):
    '''
        adds "/" to the end if needed
    '''
    if text == "": # is
        return ""
    if(text.endswith('/')):
        return text
    else:
        return text+"/"


def isEveryNPercent(current_it: int, max_it: int, percent: float = 10):
    curr_percent = current_it/max_it*100
    n = int(curr_percent/percent)

    prev_percent = (current_it-1)/max_it*100
    return ((curr_percent >= n * percent) & (prev_percent < n * percent)) or (curr_percent >= 100)


def visPointCloud(pcd, colors=None, normals=None, downsample=None, show_normals=False):
    pcd_o3 = o3d.geometry.PointCloud()
    pcd_o3.points = o3d.utility.Vector3dVector(pcd[:, 0:3])
    if colors is not None:
        pcd_o3.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        pcd_o3.normals = o3d.utility.Vector3dVector(normals)
    if downsample is not None:
        pcd_o3 = pcd_o3.voxel_down_sample(downsample)
    o3d.visualization.draw_geometries([pcd_o3], point_show_normal=show_normals)


def visPointClouds(pcd_list, colors_list=None):
    pcd_o3_list = []
    for i, pcd in enumerate(pcd_list):
        pcd_o3 = o3d.geometry.PointCloud()
        pcd_o3.points = o3d.utility.Vector3dVector(pcd[:, 0:3])
        if colors_list is not None:
            if len(colors_list) > i:
                colors = colors_list[i]
                if colors is not None:
                    pcd_o3.colors = o3d.utility.Vector3dVector(colors)
        else:
            print('have colors')

        pcd_o3_list += [pcd_o3]
    print(f'pcd_o3_list len= {len(pcd_o3_list)}')
    o3d.visualization.draw_geometries(pcd_o3_list)

# def visAll(input,)

def randomSample(nr_samples, nr_points, seed =0):
    """Samples nr_samples indices. All values in range of nr_points, no duplication

    Args:
        nr_samples ([type]): [description]
        nr_points ([type]): [description]
        seed (int, optional): [description]. Defaults to 0.
    """
    subm_idx = np.arange(nr_points)
    np.random.seed(seed)
    np.random.shuffle(subm_idx)
    # print('shuffled idx',subm_idx)
    return subm_idx[0:min(nr_points, nr_samples)]

def visVectorField(start, end, ref=None, colors=None):
    nr_p = start.shape[0]
    pcd_o3 = o3d.geometry.PointCloud()
    pcd_o3.points = o3d.utility.Vector3dVector(end[:, 0:3])
    # o3d.visualization.draw_geometries([pcd_o3])
    lines = np.concatenate((np.reshape(np.arange(nr_p), (-1, 1)),
                            np.reshape(np.arange(nr_p)+nr_p, (-1, 1))), axis=1)
    points = np.concatenate((start, end), axis=0)
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points),
                                    lines=o3d.utility.Vector2iVector(lines))
    if ref is None:
        o3d.visualization.draw_geometries([line_set, pcd_o3])
    else:
        ref_o3 = o3d.geometry.PointCloud()
        ref_o3.points = o3d.utility.Vector3dVector(ref[:, 0:3])
        if colors is not None:
            ref_o3.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([line_set, pcd_o3, ref_o3])


def renderVectorField(start, end, ref=None, colors=None, file_path='test.png'):
    nr_p = start.shape[0]
    pcd_o3 = o3d.geometry.PointCloud()
    pcd_o3.points = o3d.utility.Vector3dVector(end[:, 0:3])
    # o3d.visualization.draw_geometries([pcd_o3])
    lines = np.concatenate((np.reshape(np.arange(nr_p), (-1, 1)),
                            np.reshape(np.arange(nr_p)+nr_p, (-1, 1))), axis=1)
    points = np.concatenate((start, end), axis=0)
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points),
                                    lines=o3d.utility.Vector2iVector(lines))
    if ref is None:
        renderO3d([line_set, pcd_o3], file_path=file_path)
    else:
        ref_o3 = o3d.geometry.PointCloud()
        ref_o3.points = o3d.utility.Vector3dVector(ref[:, 0:3])
        if colors is not None:
            ref_o3.colors = o3d.utility.Vector3dVector(colors)
        renderO3d([line_set, pcd_o3, ref_o3], file_path=file_path)


def renderO3d(o3d_list, file_path='test.png'):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for o3d_g in o3d_list:
        vis.add_geometry(o3d_g)
    # vis.add_geometry(o3d_list)
    # vis.update_geometry()

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(file_path)
    # vis.run()
    vis.destroy_window()


def renderCloud(pcd, colors, file_path='test.png'):
    pcd_o3 = o3d.geometry.PointCloud()
    pcd_o3.points = o3d.utility.Vector3dVector(pcd[:, 0:3])
    if colors is not None:
        pcd_o3.colors = o3d.utility.Vector3dVector(colors)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd_o3)
    # vis.update_geometry()
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(file_path)
    # vis.run()
    vis.destroy_window()


def saveCloud2Binary(cld, file, out_path=None):
    if out_path is None:
        out_path = ''
    else:
        if not os.path.exists(out_path):
            os.makedirs(out_path)
    f = open(out_path+file, "wb")
    f.write(cld.astype('float32').T.tobytes())
    f.close()

def loadCloudFromCSV(file, cols=3):
    if (file.endswith('.csv') or file.endswith('.txt')):
        data = np.genfromtxt(file, dtype='float32', delimiter=',')
        #print("LoadCloudFromCSV" + file + "\n",my_data)
        if len(data[0]) > cols:
            for i in range(len(data[0])-1,cols-1,-1):
                #print(data[0])
                data = np.delete(data,(i), axis=1)
        #print(data[0])
        return data
    else:
        return np.array([])
	
def saveCloud2CSV(array, file):
    np.savetxt(file, array, delimiter=",")
    return 0

def loadCloudFromPLY(file, keys=["x", "y", "z"]):
    from plyfile import PlyData, PlyElement

    cloud = PlyData.read(file)
    if file.endswith(".ply"):
        if len(keys) > 0:
            row_count = 0
            for key in keys:
                if row_count == 0:
                    array = np.asarray(eval("cloud.elements[0].data[\'" + key + "\']"))
                else :
                    current_array = np.asarray(eval("cloud.elements[0].data[\'" + key + "\']"))
                    array = np.vstack((array, current_array))
                row_count += 1
        
        return array.T

def loadRawBinary(file, as_nparray=False):
    if file.endswith('.bin'):	
        f = open(file, "rb")
        binary_data = f.read()
        f.close()
        if as_nparray:
            temp = np.frombuffer(binary_data, dtype='float32', count=-1)
            return temp
        else:
            return binary_data


def loadCloudFromBinary(file, cols=5):
    file_name = file.split("/")[-1]
    """
    dir = "/".join(file.split("/")[:-1])
    
    meta = open(dir+"/meta.txt")
    meta_data = meta.readlines()
    meta.close()
    meta_array = []
    for line in meta_data:
        line = line.split("\n")[0]
        line = line.split(" : ")
        current_data = [line[0]]
        line = line[1].split(" ")
        current_data.append(line)
        meta_array.append(current_data)
    #print(meta_array)
    """
    if file.endswith('.bin'):
        #for entry in meta_array:
        #    if entry[0] == file.split(".bin")[0]:
        #        cols = len(entry[1])
    
        if len(file_name.split(".")) > 1:
            cols = int(file_name.split(".")[-2])
            #print(cols)
        else:
            cols = cols
        #print("COLS = ", cols)
        f = open(file, "rb")
        binary_data = f.read()
        f.close()
        temp = np.frombuffer(
            binary_data, dtype='float32', count=-1)
        if(not temp.size%cols):
            data = np.reshape(temp, (cols, int(temp.size/cols)))
        else:
            print("\n\nERROR : FILE of improper shape, IGNORING : " + file+"\n")
            data = np.array([])
            return None
        return data.T
    else:
        return np.array([])
		
def loadCloudFromBinBlunt(file, cols=3): #Loads even the clouds which have items not a multiple of 3 as xyz point cloud by trimming extra items
    #print(file)
    if file.endswith('.bin'):
        file_name = file.split("/")[-1]
        if len(file_name.split(".")) > 2:
            if cols == None:
                cols = int(file_name.split(".")[-2])
            #print(cols)
        else:
            cols = cols	
        f = open(file, "rb")
        binary_data = f.read()
        f.close()
        temp = np.frombuffer(
            binary_data, dtype='float32', count=-1)
        temp = temp[:(temp.size-(temp.size%cols))]
        temp = np.reshape(temp, (cols, int(temp.size/cols)))
        return temp.T
    else:
        return None

def loadCloudFromBinBluntTrans(file, cols=3, ext:str="bin"): #Loads even the clouds which have items not a multiple of 3 as xyz point cloud by trimming extra items, reads the data column-wise instead of row-wise (as in loadCLoudFromBinBlunt)
    #print(file)
    if file.endswith('.'+ext):
        file_name = file.split("/")[-1]
        if len(file_name.split(".")) > 2:
            if cols == None:
                cols = int(file_name.split(".")[-2])
            #print(cols)
        else:
            cols = cols	
        f = open(file, "rb")
        binary_data = f.read()
        f.close()
        temp = np.frombuffer(
            binary_data, dtype='float32', count=-1)
        temp = temp[:(temp.size-(temp.size%cols))] # Removes extra entries
        new_temp = []
        for i in range(cols):
            for item_count in range(len(temp)):
                if not (item_count-i)%cols:
                    new_temp.append(temp[item_count])
        temp = np.array(new_temp)
        temp = np.reshape(temp, (cols, int(temp.size/cols)))
        return temp.T
    else:
        return None

def colorizeConv(in_pcl:np.ndarray, out_pcl:np.ndarray, kernel_radius, max_nr_neighbors,kernel_pos=None, kernel_points=None):
    import octree_handler

    octree = octree_handler.Octree()
    octree.setInput(in_pcl)
    out_index = random.randrange(out_pcl.shape[0])
    in_index = octree.radiusSearchPoints(
        out_pcl[out_index:out_index+1, :], max_nr_neighbors, kernel_radius)
    in_index = in_index[in_index < in_pcl.shape[0]]
    in_clr = np.zeros_like(in_pcl)
    in_clr[in_index, :] = np.ones_like(
        in_pcl[in_index, :])*np.array([1, 0, 0])

    out_clr = np.zeros_like(out_pcl)
    out_clr[out_index, :] = np.array([1, 0, 0])
    
    if kernel_pos is not None:
        out_pt = out_pcl[out_index:out_index+1,:]
        k_in = kernel_pos[out_index,:,:]+out_pt
        k_clr = np.ones_like(k_in)*np.array([0, 1, 0])
        print('kernel_i coords ',kernel_pos[out_index,:,:])
        print('kernel_i coords ',kernel_points)
        print(f'in {in_pcl.shape},out {out_pcl.shape},kernel_def {kernel_pos.shape},kernel_def_i {k_in.shape},kernel {kernel_points.shape}',)
        in_pcl = np.vstack((in_pcl,k_in))
        in_clr = np.vstack((in_clr,k_clr))

    if kernel_points is not None:
        out_pt = out_pcl[out_index:out_index+1,:]
        k_in = kernel_points+out_pt
        k_clr = np.ones_like(k_in)*np.array([0, 0, 1])
        in_pcl = np.vstack((in_pcl,k_in))
        in_clr = np.vstack((in_clr,k_clr))
    # visPointClouds([in_pcl,out_pcl+1],[in_clr,out_clr])
    # if kernel_pos is not None:
    

    return (in_pcl, out_pcl), (in_clr, out_clr)

# def colorizeDeformedKP(in_pcl, out_pcl, kernel_radius, max_nr_neighbors,kern_pcl=None):
#     octree = octree_handler.Octree()
#     octree.setInput(in_pcl)

def visualizeConv(in_out_pts, in_out_clr):
    """[summary]

    Arguments:
        in_out_pts [list] -- [(in_pcl1,out_pcl1),(in_pcl2,out_pcl2),...] pointclouds 
        in_out_clr [list] -- [(in_clr1,out_clr1),(in_clr2,out_clr2),...] colors
    """
    extend = 1.2
    pts = []
    clrs = []
    n = len(in_out_pts)
    for i in range(n):
        in_pcl, out_pcl = in_out_pts[i]
        row = np.array([1, 0, 0])*extend * i
        col = np.array([0, -1, 0])*extend
        pts.append(in_pcl+row)
        pts.append(out_pcl+row+col)
        clrs.append(in_out_clr[i][0])
        clrs.append(in_out_clr[i][1])
    visPointClouds(pts, clrs)

# https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file/32216025


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def findList(inp_list, value):
    for i, item in enumerate(inp_list):
        # print( item,value)
        if item == value:
            return i
def cloud_is_supported(fileName):
    ext = fileName.split(".")[-1]
    supportedFormats = ["csv", "bin", "txt"]
    
    if ext in supportedFormats:
        return True
    else :
        return False

def listFiles(path):
    fileList = []
    for (root, dirs, files) in os.walk(path):
           for i in range(len(files)):
               files[i] = path+"/"+files[i]
           fileList += files
           fileList.sort()
    return fileList
    
def loadCloud(path=None, mode="fl"):
    import dialogs as dia

    if path == None:
        if mode != "fld":
            path = dia.selectFile()
        if mode == "fld":
            path = dia.selectFolder()
            print(path)
    if path !=None and (os.path.isdir(path) or mode == "fld"):
        fileList = listFiles(path)
        """
        fileList = []
        for (root, dirs, files) in os.walk(path):
           for i in range(len(files)):
               files[i] = path+"/"+files[i]
           fileList += files
        """
        print(fileList)
        #assert len(fileList)==1 # Check that the fileList is actually a list within list
        #fileList = fileList[0]
        cloud_array = mergeClouds(fileList)
    elif cloud_is_supported(path):
        ext = path.split(".")[-1]
        if ext == "bin":
            cloud_array = loadCloudFromBinBlunt(path)
        elif (ext == "csv" or ext == "txt"):
            cloud_array = loadCloudFromCSV(path)
    return cloud_array

def downSampleCloud(array, points="10000"): # Downsample the point cloud to a given max number of points
    total_points = len(array)
    if points.endswith("%"):
        points = total_points*(int(points.split("%")[0])/100)
    else :
        points = int(points)

    assert (points > 0)
	
    down_perc = points*100/total_points
    assert down_perc <= 100
    new_cloud = None
    if input("Down-sampling mode : random? (y/n)") == "y":
        for i in tqdm(range(total_points)):
            if i == 0:
                new_cloud = np.array(array[i])
                continue
            else:
                select = np.random.randint(0,100)
                if select <= down_perc:
                    new_cloud = np.vstack([new_cloud,array[i]])
    else:
        step = int(total_points/points)
        for i in tqdm(range(0,total_points,step)):
            if i == 0:
                new_cloud = np.array(array[i])
                continue
            else:
                new_cloud = np.vstack([new_cloud,array[i]])
    print("\nDOWN-SAMPLE CHECK : Points [new cloud / original cloud] = [ " + str(len(new_cloud)) + " / " + str(total_points) + " ] = " + str(len(new_cloud)*100/total_points) + " %")
    return new_cloud			

def downSampleCSVCloud(file=None, points="10000", out_file=temp_loc+"/pcu.downSampleCSV.csv"):	
    import dialogs as dia

    if file == None:
        file = dia.selectFile()
    array = loadCloudFromCSV(file)
    total_points = len(array)
    
    del array # Delete array from memory (local scope)
    
    if points.endswith("%"):
        points = total_points*(int(points.split("%")[0])/100)
    else :
        points = int(points)

    assert (points > 0)
	
    down_perc = points*100/total_points
    assert down_perc <= 100
    
    print("\n\nWARN : Random downsampling is not supported.\n\n")

    step = int(total_points/points)
    with open(file) as infile:
        i=0 # Counts points in originak cloud
        point_count = 0 # Counts points in new cloud
        iter_list = list(np.arange(0,total_points,step))
        for line in tqdm(infile):
            if i == 0:
                out = open(out_file, "a")
                if i == 0:
                    #TODO : Delete any existing out_file
                    pass
                out.write(line)
                out.close()
                point_count += 1
            if i == step-1:
                i=-1
            i+=1

    print("\nDOWN-SAMPLE CHECK : Points [new cloud / original cloud] = [ " + str(point_count) + " / " + str(total_points) + " ] = " + str(point_count*100/total_points) + " %")


	  
def mergeClouds(cloudsLocList=[]):
    for i in tqdm(range(len(cloudsLocList))):
        cloudLoc = cloudsLocList[i]
        if not cloud_is_supported(cloudLoc):
            continue
        elif i == 0:
            cloud = loadCloud(cloudLoc, mode="fl")
        else:
            try:
                cloud = np.vstack((cloud,loadCloud(cloudLoc, mode="fl")))
            except:
                print("Error in vstacking, ignoring file : " + cloudLoc, loadCloud(cloudLoc, mode="fl").shape)
    print(cloud)
    if input("Points in current cloud = %d.\n Do you wish to downsample the cloud (y/n)" %(len(cloud))) == "y":
        points = int(input("Please enter the number of points you want to downsample to : "))
        cloud = downSampleCloud(array=cloud, points=points)
    if input("Do you wish to save the merged point cloud as CSV? (y/n)") == "y":
        saveCloud2CSV(cloud, "merged-cloud.csv")
    return cloud
       
def previewCSV(file=None, rows=10, cols=3):
    import dialogs as dia

    if file == None:
        file = dia.selectFile()
    with open(file) as infile: # Reading the file one line at a time, without loading entire file into memory
        line_counter = 0
        for line in infile:
            if line_counter > 5:
                break
            print(line)
            line_counter += 1
    
def trimCSV(mode="array", in_file=None, out_file=temp_loc+"/pcu.trimCSV.csv", start_row=0, end_row=-1, start_col=0, end_col=5, rows=[], cols=[]): # This function trims the CSV file and returns a numoy array. We can use Pandas instead?
    import dialogs as dia

    if in_file == None:
        in_file = dia.selectFile()
    with open(in_file) as infile: # Reading the file one line at a time, without loading entire file into memory
        line_counter = 0
        current_array = []
        array = np.array([])
        
        for line in infile:
            if (line_counter >= start_row and (end_row == -1 or line_counter <= end_row)):
                if line.endswith("\n"):
                    line = "\n".join(line.split("\n")[0:-1])
                #if line_counter > 5:
                #    break
                current_array = line.split(",")
                if mode == "array":
                    current_array = np.asarray(current_array, dtype=np.float32)[start_col:end_col]
                    if len(array) == 0:
                        array = current_array
                    else : 
                        #print(array)
                        #print(current_array)
                        array = np.vstack((array,current_array))
                    #print(current_array, type(current_array))
                elif mode == "append_csv":
                    line = ",".join(line.split(",")[start_col:end_col])
                    if not line.endswith("\n"):
                        line += "\n"
                    file = open(out_file, "a")
                    file.write(line)
                    file.close()
                    
            line_counter += 1
        print(array)
    pass #TODO

def pc_stats(file=None):
    import dialogs as dia

    if file == None:
        file = dia.selectFile() #"/home/ax/axcore/ms-project/data/dales/ply/train/5080_54435.ply"
    print("File : %15s" %file)
    format = file.split(".")[-1].upper()
    print("Format : %15s"  %format, end="\n")

    if format == "PLY":
        loadCloudFromPLY(file)

    if format == "BIN":
        unshaped_bin = loadRawBinary(file, as_nparray=True)
        print("Number of data items : %15d" %len(unshaped_bin))
        possible_cols = []
        for i in range(1,21):
            if not (len(unshaped_bin)% i):
                possible_cols.append(i)
        print("Possible no of columns : ", possible_cols)
        
        #print(unshaped_bin)
        start = None
        end = None
        header_exists = np.isnan(unshaped_bin[0])
        print("Headers Found : %r" %(header_exists))
        if header_exists:
            print("Detecting shape from headers...\n")
            for i in tqdm(range(len(unshaped_bin))):
                if np.isnan(unshaped_bin[i]):
                    if (start == None):
                        start = i
                    elif (end == None):
                        end = i
                if(start != None and end != None):
                    break
            rows = end-start
            print("Possible rows : %15d" %i)
            print("Number of points : %d" %(rows-1))
            print("Possible columns shape test passed : %15r" %(not bool(len(unshaped_bin)%rows)))
            if not bool(len(unshaped_bin)%rows):
                # Possible Headers
                unshaped_raw_binary = loadRawBinary(file, as_nparray=False)
                headers = []
                for i in range(0,len(unshaped_bin),rows):
                    print(unshaped_raw_binary[i],unshaped_raw_binary[i+1],i)

            # Figure out the header
        pass
    elif format == "CSV":
        array = loadCloudFromCSV(file)
        print("Number of points : %15d" %len(array))
        print("Preview : \n")
        print(array[:5], end="\n=======Preview End========\n\n")

#trimCSV(mode="append_csv", start_row=0, end_row=-1, start_col=0, end_col=3)    

def subdivide_cloud(input_cloud=None, shape:list=[2,2], ply_keys:list=[]):
    import dialogs as dia

    if not input_cloud:
        print("Please select an input cloud")
        input_cloud = dia.selectFile()

    format = input_cloud.split(".")[-1]

    sub_clouds = []

    if format == "ply":
        if not ply_keys:    
            attribute_list = input("Please enter the comma-separated list of attributes present in the point cloud : ").split(",")
        else:
            attribute_list = ply_keys
        cloud = loadCloudFromPLY(input_cloud, keys=attribute_list)
    elif format == "txt" or format == "csv":
        cloud = loadCloudFromCSV(input_cloud)
    elif format == "bin":
        attribute_number = int(input("Please enter the number of attributes in the point cloud (integer) : "))
        cloud = loadCloudFromBinBlunt(file=input_cloud, cols=attribute_number)

    # Perform analysis of the x, y, z ranges of the cloud.
    x_range = [0,0]
    y_range = [0,0]
    z_range = [0,0]

    ranges = [x_range, y_range, z_range] # Format for each coordinate = min, max

    for point in tqdm(cloud):
        coordinates = point[:2]

        for index in range(len(ranges)):
            coordinate = coordinates[index]
            print("These will now be compared...",type(ranges[0][0]), type(coordinates[0]))

            
            if ranges[index][0] == 0 and ranges[index][1] == 0:
                ranges[index][0] = coordinate
                ranges[index][1] = coordinate
            if float(ranges[index][0]) > float(coordinate):
                ranges[index][0] = coordinate
            elif float(ranges[index][1]) < float(coordinate):
                ranges[index][1] = coordinate
            
    
    print("Cloud subdivision complete the x, y, z ranges are as follows : " + str(x_range) + " | " + str(y_range) + " | " + str(z_range))
            
    # Find out the x, y, z ranges to perform the sub-divisions at.

    # Perform the subdivide operation

    return sub_clouds


if __name__ == "__main__":
    a = np.random.rand(12, 3)
    
    subdivide_cloud(input_cloud="/DATA/aakash/paper-1/skitti/c0/000000.ply", ply_keys=["x", "y", "z", "scalar_Scalar_field", "scalar_Scalar_field"])
    
    # saveCloud2Binary(a,'test.bin')
    # b = loadCloudFromBinary('test.bin')
    # print(a-b)
    # cld = loadCloudFromBinary('/media/lwiesmann/WiesmannIPB/data/data_kitti/dataset/submaps/04/000004.bin')
    # visPointCloud(cld)
    # start = np.array([[0, 0, 0],
    #                   [0, 1, 0],
    #                   [0, 0, 1]])

    # end = np.array([[0, 0, 0],
    #                 [1, 0, 0],
    #                 [0, 0, 1]])+2
    # visVectorField(start, end)
    # renderCloud(a)
