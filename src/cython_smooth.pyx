import numpy 
cimport numpy 
cimport cython
import math

ctypedef numpy.uint8_t DTYPE_t

# Function to manually smooth image using median filter
cpdef manual_smooth(numpy.ndarray[DTYPE_t, ndim=2] arr_img, int kernel_size):
    cdef int rows
    cdef int columns
    cdef int i,j,pad_num,start
    
    pad_num = (kernel_size-1)
    start = numpy.int(((kernel_size+1)/2))-1
    
    rows = arr_img.shape[0]
    columns = arr_img.shape[1]
    
    cdef numpy.ndarray[numpy.uint8_t, ndim=2] new_arr
    # New array with zero padding
    new_arr = numpy.zeros([rows+pad_num, columns+pad_num],dtype = numpy.uint8)
    
    #for i in range(start,new_arr.shape[0]-start):
        #for j in range(start,new_arr.shape[1]-start):
            # assigning pixel values in new array from original array
            #new_arr[i,j] = arr_img[i-start,j-start] 
    
    # assigning pixel values in new array from original array
    new_arr[start:new_arr.shape[0]-start,start:new_arr.shape[1]-start] = arr_img

    # Convolving with median filter
    for i in range(kernel_size,new_arr.shape[0]+1):
        for j in range(kernel_size,new_arr.shape[1]+1):
            arr_img[i-kernel_size,j-kernel_size] = numpy.median(new_arr[i-kernel_size:i,j-kernel_size:j]) 
            
    return arr_img


# Function to calculate derivative of the image
cpdef derivative(numpy.ndarray[DTYPE_t, ndim=2] arr_img,numpy.ndarray[DTYPE_t, ndim=2] mask):
    cdef int rows
    cdef int columns
    cdef int i,j
    
    rows = arr_img.shape[0]
    columns = arr_img.shape[1]
    
    cdef numpy.ndarray[numpy.uint8_t, ndim=2] new_arr
    # New array with zero padding
    new_arr = numpy.zeros([rows+2, columns+2],dtype = numpy.uint8)
    
    for i in range(1,new_arr.shape[0]-1):
        for j in range(1,new_arr.shape[1]-1):
            # assigning pixel values in new array from original array
            new_arr[i,j] = arr_img[i-1,j-1] 
            
    cdef int sum1 
    # Convolving with derivative filter
    for i in range(1,new_arr.shape[0]-1):
        for j in range(1,new_arr.shape[1]-1):
            sum1 = 0
            sum1 += (new_arr[i-1,j-1]*mask[0,0])
            sum1 += (new_arr[i-1,j]*mask[0,1])
            sum1 += (new_arr[i-1,j+1]*mask[0,2])
            sum1 += (new_arr[i,j-1]*mask[1,0])
            sum1 += (new_arr[i,j]*mask[1,1])
            sum1 += (new_arr[i,j+1]*mask[1,2]) 
            sum1 += (new_arr[i+1,j-1]*mask[2,0])
            sum1 += (new_arr[i+1,j]*mask[2,1])
            sum1 += (new_arr[i+1,j+1]*mask[2,2]) 

            arr_img[i-1,j-1] =  sum1
            
    # Normalize the values between [0,255]
    for i in range(arr_img.shape[0]):
        for j in range(arr_img.shape[1]): 
            if(arr_img[i,j] < 0):
                arr_img[i,j] = 0
            else:
                arr_img[i,j] = arr_img[i,j] % 256 
            
    return arr_img

# Function to calculate magnitude of gradient of the image
cpdef magn_gradient(numpy.ndarray[DTYPE_t, ndim=2] gradx,numpy.ndarray[DTYPE_t, ndim=2] grady):
    cdef int rows
    cdef int columns
    cdef int i,j
    
    rows = gradx.shape[0]
    columns = gradx.shape[1]
    
    cdef numpy.ndarray[numpy.uint8_t, ndim=2] mag_arr
    # New array with zero padding
    mag_arr = numpy.zeros([rows, columns],dtype = numpy.uint8)
    
    for i in range(gradx.shape[0]):
        for j in range(gradx.shape[1]):
            mag_arr[i,j] = (math.sqrt(gradx[i,j]**2 + grady[i,j]**2))%256
             
            
    return mag_arr
