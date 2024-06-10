import numpy as np
import skimage
import tifffile

def predict_mask(model, image_path, image_size, patch_size):
    latticeMovieImage = tifffile.imread(image_path)
    latticeMovieImage = latticeMovieImage[0,:image_size, :image_size, :image_size]
    result = np.zeros((image_size, image_size, image_size, 1))

    frames = latticeMovieImage.shape[0]
    for frame in range(frames):

        for x in range(image_size//patch_size):
            for y in range(image_size//patch_size):
                for z in range(image_size//patch_size):
                    x_index = x*patch_size
                    y_index = y*patch_size
                    z_index = z*patch_size
                
                    current_lattice_patch = latticeMovieImage[frame,z_index:z_index+patch_size, y_index:y_index+patch_size, x_index:x_index+patch_size]
                    current_lattice_patch = current_lattice_patch.reshape(1, patch_size, patch_size, patch_size, 1)
                
                    result_patch = model.predict(current_lattice_patch)
                    for i in range(patch_size):
                        for j in range(patch_size):
                            for k in range(patch_size):
                                result_pixel = result_patch[0, i, j, k, 0]
                                result[z_index+i, y_index+j, x_index+k, 0] = result_pixel
                        
    return result.reshape(1, image_size, image_size, image_size, 1)

##NEED TO CORRECT THIS FUNCTION
#Automatically trim image of rectangular shape and generate patches
def predict_mask(model, image_path, patch_size, offset=np.zeros((3,), dtype=int)):
    assert offset.size is 3, "Offset array needs to have a size of 3."
    
    latticeMovieImage = tifffile.imread(image_path)

    frames = latticeMovieImage.shape[0]

    for frame in range(frames):
        x_extra = latticeMovieImage[frame].shape[2]%patch_size
        x_size = latticeMovieImage[frame].shape[2] - x_extra
        if offset[0] > x_extra:
            print("1st dim offset exceeds image dim")
            offset[0] = 0
        
        y_extra = latticeMovieImage[frame].shape[1]%patch_size
        y_size = latticeMovieImage[frame].shape[1] - y_extra
        if offset[1] > y_extra:
            print("2st dim offset exceeds image dim")
            offset[1] = 0
        
        z_extra = latticeMovieImage[frame].shape[0]%patch_size
        z_size = latticeMovieImage[frame].shape[0] - z_extra
        if offset[2] > z_extra:
            print("3rd dim offset exceeds image dim")
            offset[2] = 0
        
        latticeMovieImage_frame = latticeMovieImage[frame, offset[0]:z_size+offset[0], offset[1]:y_size+offset[1], offset[2]:x_size+offset[2]]
        print("Image cropped to: " + str(x_size) + ", " + str(y_size) + ", " + str(z_size))
        latticeMovieImage_all.append(latticeMovieImage_frame)
        
    
        latticeMovieImage_all =  np.array(latticeMovieImage_all)

        result = np.zeros((frames, z_size, y_size, x_size, 1))

        for frame in range(frames):
            for x in range(x_size // patch_size):
                for y in range(y_size // patch_size):
                    for z in range(z_size // patch_size):
                        x_index = x * patch_size
                        y_index = y * patch_size
                        z_index = z * patch_size

                        current_lattice_patch = latticeMovieImage_all[frame, z_index:z_index + patch_size, y_index:y_index + patch_size,
                                                x_index:x_index + patch_size]
                        current_lattice_patch = current_lattice_patch.reshape(1, patch_size, patch_size, patch_size, 1)

                        result_patch = model.predict(current_lattice_patch)
                        for i in range(patch_size):
                            for j in range(patch_size):
                                for k in range(patch_size):
                                    result_pixel = result_patch[0, i, j, k, 0]
                                    result[frame,z_index + i, y_index + j, x_index + k, 0] = result_pixel

    return result.reshape(1, z_size, y_size, x_size, 1)