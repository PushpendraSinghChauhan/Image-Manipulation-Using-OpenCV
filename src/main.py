import sys
import cv2
import os
import numpy as np
import cython_smooth
import math
import matplotlib.pyplot as plt

if __name__ == '__main__':

    if(len(sys.argv) == 2):
        input_file = sys.argv[1]
        filename = "../data/" + input_file
        image = cv2.imread(filename)
        if (image is None):
            print("File with name - \'"+input_file+"\' does not exist in data folder .")
            inp_file = input("PRESS - 'C' to capture image using Camera  OR  Press - 'D' to continue with default image ")
            if(inp_file == 'D'):
                filename = "../data/house3.jpg"
                image = cv2.imread(filename)
        
            elif(inp_file == 'C'):
                camera = cv2.VideoCapture(0)
                input("Press Enter to capture !")
                return_value, image = camera.read()
                # Handle large images
                if(image.shape[0]>800 or image.shape[1]>500):
                    filename = "../data/resizeImage.jpg"
                    image = cv2.resize(image,(800,500))
                    cv2.imwrite(filename, image)                
                camera.release()
            
            else:
                sys.exit("Please input correct filename from command line again !") 
        
        else:
            # Handle large images
            if(image.shape[0]>800 or image.shape[1]>500):
                filename = "../data/resizeImage.jpg"
                image = cv2.resize(image,(800,500))
                cv2.imwrite(filename, image)
                
        
    else:
        print("No Image file name entered .")
        inp_file = input("PRESS - 'C' to capture image using Camera  OR  Press - 'D' to continue with default image ")
        if(inp_file == 'D'):
            filename = "../data/house3.jpg"
            image = cv2.imread(filename)
        
        elif(inp_file == 'C'):
            camera = cv2.VideoCapture(0)
            input("Press Enter to capture !")
            return_value, image = camera.read()
            # Handle large images
            if(image.shape[0]>800 or image.shape[1]>500):
                filename = "../data/resizeImage.jpg"
                image = cv2.resize(image,(800,500))
                cv2.imwrite(filename, image)
            camera.release()
            
        else:
            sys.exit("Please input correct filename from command line again !") 
    
    def cycle_channel(img):
        # Corresponding channels are seperated
        B, G, R = cv2.split(img)
  
        cv2.imshow("blue", B)
        cv2.waitKey(0)
        #os.system("pause") 
  
        cv2.imshow("Green", G) 
        cv2.waitKey(0)
        #os.system("pause") 
  
        cv2.imshow("red", R) 
        cv2.waitKey(0)
        #os.system("pause") 
  
        cv2.destroyAllWindows()
    
    def manual_grayscale(img):
        # Using formula from Wikipedia : gray = R * 0.2126 + G * 0.7152 + B * 0.0722 . Image is read in BGR format in openCV
        B, G, R = cv2.split(img)
        img1 = (B*0.0722 + G*0.7152 + R*0.2126).astype(np.uint8)
        return img1   
    
    def manual_smooth(img,kernel):
        # Calling cython function 'manual_smooth'
        image1 = cython_smooth.manual_smooth(img,kernel)
        return image1.astype(np.uint8) 
    
    def x_derivative(img):
        x_derivative_filter = np.array([[-1,0,1],[-1,0,1],[-1,0,1]]).astype(np.uint8)
        # Calling cython function 'derivative'
        image1 = cython_smooth.derivative(img,x_derivative_filter)
        return image1.astype(np.uint8)
    
    def y_derivative(img):
        y_derivative_filter = np.array([[1,1,1],[0,0,0],[-1,-1,-1]]).astype(np.uint8)
        # Calling cython function 'derivative'
        image1 = cython_smooth.derivative(img,y_derivative_filter)
        return image1.astype(np.uint8)
    
    def track_N(val):
        pass
        
        
    
    
    while(True):
        inp = input("Input from = ['i','w','g','G','c','s','S','d','D','x','y','m','p','r']. Input 'h' for help. Input 'exit' to exit\n")
        inp = inp.strip()
    
        # Reload the original image
        if(inp == "i"):
            image = cv2.imread(filename)
            cv2.imshow("Original Image Reloaded",image)
            cv2.waitKey(0)
            #os.system("pause")
            cv2.destroyAllWindows()

        # Save the current image into the file "out.jpg"
        elif(inp == "w"):
            cv2.imwrite("..\data\out.jpg",image)
            print("Current processed image is successfully saved in 'data' folder with name 'out.jpg'")

        # Convert the image to grayscale using the openCV conversion function
        elif(inp == "g"):
            image = cv2.imread(filename)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            cv2.imshow("Grayscale Image",image)
            cv2.waitKey(0)
            #os.system("pause")
            cv2.destroyAllWindows()

    
        # Convert the image to grayscale using your implementation of conversion function
        elif(inp == "G"):
            image = cv2.imread(filename)
            # Calling function
            image = manual_grayscale(image)
            cv2.imshow("Manual Grayscale Image",image)
            cv2.waitKey(0)
            #os.system("pause")
            cv2.destroyAllWindows()            
        
        # Cycle through the color channels of the image showing a different channel every time the key is pressed
        elif(inp == 'c'):
            image = cv2.imread(filename)
            # Calling function
            cycle_channel(image) 
            
        # Convert image to grayscale and smooth it using the openCV function. Use track bar to control the amount of smoothing 
        elif(inp == 's'):
            image = cv2.imread(filename)
            # Convert to grayscale
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            
            title_window = "Grayscale and Smoothing"
            cv2.namedWindow(title_window)
            # create trackbars for controlling N
            cv2.createTrackbar('Smooth',title_window,0,20,track_N) 
            image1 = image.copy()
            num_loop = 0
            while(True):
                num_loop += 1
                cv2.imshow(title_window,image1)
                cv2.waitKey(0)
                if(num_loop == 5):
                    break
                            
                size = cv2.getTrackbarPos('Smooth',title_window)  
                kernel = 2*size+1
                image1 = cv2.GaussianBlur(image,(kernel,kernel),0)
                
            cv2.destroyAllWindows()            

    
        # Convert image to grayscale and smooth it using your function which should perform convolution with a suitable filter. 
        # Use track bar to control the amount of smoothing 
        elif(inp == 'S'):
            image = cv2.imread(filename)
            # Calling function for manual grayscale
            image = manual_grayscale(image)
            
            title_window = "Manual Grayscale and Smoothing"
            cv2.namedWindow(title_window)
            # create trackbars for controlling N
            cv2.createTrackbar('Smooth',title_window,0,10,track_N) 
            image1 = image.copy()
            num_loop = 0
            while(True):
                num_loop += 1
                cv2.imshow(title_window,image1)
                cv2.waitKey(0)
                if(num_loop == 5):
                    break
                            
                size = int(cv2.getTrackbarPos('Smooth',title_window) ) 
                kernel = 2*size+1
                # Calling function for manual smoothing
                if(size >= 1):
                    image1 = manual_smooth(image.copy(),kernel)
                else:
                    image1 = image.copy()
                
            cv2.destroyAllWindows()            
        
        # Downsample the image by a factor of 2 without smoothing
        elif(inp == 'd'):
            image = cv2.imread(filename)
            # Code to downsample by factor of 2
            image = cv2.resize(image, (image.shape[0]//2,image.shape[1]//2))
            cv2.imshow("Downsample Image by 2 Without Smoothing",image)
            cv2.waitKey(0)
            #os.system("pause")
            cv2.destroyAllWindows()            
        
        # Downsample the image by a factor of 2 with smoothing
        elif(inp == 'D'):
            image = cv2.imread(filename)
            # Code to downsample by factor of 2
            image = cv2.resize(image, (image.shape[0]//2,image.shape[1]//2))
            # Code to smooth the image using gaussian filter
            image = cv2.GaussianBlur(image,(5,5),0)
            cv2.imshow("Downsample Image by 2 With Smoothing",image)
            cv2.waitKey(0)
            #os.system("pause")
            cv2.destroyAllWindows()
        
        # Convert the image to grayscale and perform convolution with an X derivative filter.
        # Normalize the obtained values to the range [0,255]
        elif(inp == "x"):
            image = cv2.imread(filename)
            # Convert image to grayscale
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            # Calling function for x-derivative
            image = x_derivative(image)
            cv2.imshow("Grayscale Image and X-derivative",image)
            cv2.waitKey(0)
            #os.system("pause")
            cv2.destroyAllWindows()
        
        # Convert the image to grayscale and perform convolution with a y derivative filter.
        # Normalize the obtained values to the range [0,255]
        elif(inp == "y"):
            image = cv2.imread(filename)
            # Convert the image to grayscale
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            # Calling function for y-derivative
            image = y_derivative(image)            
            cv2.imshow("Grayscale Image and Y-derivative",image)
            cv2.waitKey(0)
            #os.system("pause")
            cv2.destroyAllWindows()
        
        # Show the magnitude of the gradient normalized to the range [0,255]. The gradient is computed
        # based on the X and Y derivatives of the image
        elif(inp == "m"):
            image = cv2.imread(filename)
            # Convert image to grayscale
            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            # Calling function for x-derivative
            grad_x = x_derivative(gray)
            # Calling function for y-derivative
            grad_y = y_derivative(gray)
            # Calling the function 'magn_gradient' of cython_smooth to find magnitude of the gradient
            image = cython_smooth.magn_gradient(grad_x,grad_y)
            image = image.astype(np.uint8)
            cv2.imshow("Magnitude of the gradient",image)
            cv2.waitKey(0)
            #os.system("pause")
            cv2.destroyAllWindows()            
        
        # Convert the image to grayscale and plot the gradient vectors of the image every N pixels and let the plotted gradient
        # vectors have a length of K. Use a track bar to control N. Plot the vectors as short line segments of length K
        elif(inp == "p"):
            image = cv2.imread(filename)
            # Convert image to grayscale
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            # Calling function for x-derivative
            grad_x = x_derivative(image)
            # Calling function for y-derivative
            grad_y = y_derivative(image)            
            
            slider_max = int(image.shape[0]*image.shape[1])
            title_window = "Gradient vectors for N pixels"
            cv2.namedWindow(title_window)
            # create trackbars for controlling N
            cv2.createTrackbar('N',title_window,0,slider_max,track_N)
            num_loop = 0
            while(True):
                num_loop += 1
                cv2.imshow(title_window,image)
                cv2.waitKey(0)
                if(num_loop == 5):
                    break
                            
                N = cv2.getTrackbarPos('N',title_window)
                X_loc = np.array(([0]*N),dtype=int)
                Y_loc = np.array(([0]*N),dtype=int)
                U_dir = np.array(([0]*N),dtype=int)
                V_dir = np.array(([0]*N),dtype=int)
                n_pixels = 0
                for i in range(grad_x.shape[0]):
                    for j in range(grad_x.shape[1]):
                        
                        X_loc[n_pixels] = i
                        Y_loc[n_pixels] = j
                        U_dir[n_pixels] = grad_x[i,j]
                        V_dir[n_pixels] = grad_y[i,j]
                        n_pixels += 1
                        if(n_pixels == N):
                            break
                    
                    if(n_pixels == N):
                        break
                fig, ax = plt.subplots(figsize=(12,10))
                q = ax.quiver(X_loc,Y_loc,U_dir,V_dir,minlength=3)
                ax.quiverkey(q, X=0.3, Y=1.1, U=10,label='Gradient Vectors for N pixels', labelpos='E')                        
                plt.xlabel("Grad_X")
                plt.ylabel("Grad_Y")
                temp_path = "../data/gradient_vector.jpg"
                plt.savefig(temp_path)
                image = cv2.imread(temp_path)
                image = cv2.resize(image,(900,800))
                
            cv2.destroyAllWindows()    
        
        # convert the image to grayscale and rotate it using an angle of θ degrees. Use a track bar to control the rotation angle
        elif(inp == "r"):
            image = cv2.imread(filename)
            # Convert image to grayscale
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            
            title_window = "Grayscale Image Rotation"
            cv2.namedWindow(title_window)
            # create trackbars for controlling N
            cv2.createTrackbar('Angle',title_window,0,360,track_N)
            rows = image.shape[0]
            cols = image.shape[1]
            image1 = image.copy()
            num_loop = 0
            while(True):
                num_loop += 1
                cv2.imshow(title_window,image1)
                cv2.waitKey(0)
                if(num_loop == 5):
                    break
                            
                angle = cv2.getTrackbarPos('Angle',title_window)  
                M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
                image1 = cv2.warpAffine(image,M,(cols,rows))
                
            cv2.destroyAllWindows()    
        
        # Display a short description of the program, its command line arguments and the keys it supports .
        elif(inp == "h"):
            # Code to do
            print("\nPROGRAM DESCRIPTION : In this program, simple image manipulation is being performed using openCV ")
            print("and my own implementation of some conversion functions")
            print("\n COMMAND LINE ARGUMENT - (1). python main.py 'image_filename.jpg'  (2). python main.py")
            print("\n SUPPORTED KEYS:-")
            print("(1). 'i' - Reload the original image")
            print("(2). 'w' - Save the current image into the file 'out.jpg'")
            print("(3). 'g' - Convert the image to grayscale using openCV conversion function")
            print("(4). 'G' - Convert the image to grayscale using my own implementation of conversion function")
            print("(5). 'c' - Cycle through the color channels of the image showing a different channel everytime the key is pressed")
            print("(6). 's' - Convert the image to grayscale and smooth it using the openCV function")
            print("(7). 'S' - Convert the image to grayscale and smooth it using my own implementation function")
            print("(8). 'd' - Downsample the image by a factor of 2 without smoothing")
            print("(9). 'D' - Downsample the image by a factor of 2 with smoothing")
            print("(10). 'x' - Convert the image to grayscale and perform convolution with an X-derivative filter")
            print("(11). 'y' - Convert the image to grayscale and perform convolution with a Y-derivative filter")
            print("(12). 'm' - Show the magnitude of the gradient normalized to the range [0,255]")
            print("(13). 'p' - Convert the image to grayscale and plot the gradient vectors of the image every N pixels")
            print("(14). 'r' - Convert the image to grayscale and rotate it using an angle of θ degrees")
            print("(15). 'h' - Display a short description of the program, its command line arguments, and the keys it supports")
            print("(16). 'exit' - Exit the program\n")
            
        
        # Exit the program
        elif(inp == "exit"):
            break
    
        else:
            print(" Please input valid entry . For more info please input 'h' ")
        
        

            