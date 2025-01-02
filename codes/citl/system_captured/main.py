import os
import time
import PySpin
import imageio
import detect_heds_module_path
import holoeye
from holoeye import slmdisplaysdk
import numpy as np
NUM_IMAGES = 1000  # Number of images to capture
displayOptions = slmdisplaysdk.ShowFlags.PresentAutomatic
from PIL import Image

def acquire_images(cam, nodemap, nodemap_tldevice, slm):

    print('*** IMAGE ACQUISITION ***\n')
    try:
        result = True

        # Set camera acquisition mode to Continuous
        node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
        if not PySpin.IsReadable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
            print('Unable to set acquisition mode to Continuous (enumeration retrieval). Aborting...')
            return False

        node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
        if not PySpin.IsReadable(node_acquisition_mode_continuous):
            print('Unable to set acquisition mode to Continuous (entry retrieval). Aborting...')
            return False

        acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()
        node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

        print('Acquisition mode set to Continuous...')

        for i in range(2700):
            print(i)
            try:
                # Load image on SLM
                image_path = 'C:/Users/10735/Desktop/ampshift/2/%04d.png' % (i)
                error = slm.showDataFromFile(image_path, displayOptions)
                assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)
                # print('Image loaded on SLM: %s' % image_path)

                # Wait for image stabilization
                time.sleep(0.4)

                # Capture image with camera
                result = None
                # start_time = time.time()
                for j in range(5):
                    # Capture image with camera
                    cam.BeginAcquisition()
                    image_result = cam.GetNextImage(1000)
                    image_data = np.array(image_result.GetNDArray())  # Convert image to array
                    image_result.Release()
                    cam.EndAcquisition()

                    # Accumulate image data
                    if result is None:
                        result = image_data.astype(np.float32)
                    else:
                        result += image_data
                # print(time.time() - start_time)

                result /= 5
                result = result.astype(np.uint8)
                # Save the final result
                filename = 'C:/Users/10735/Desktop/0405/2/%04d.png' % (i)
                final_result = Image.fromarray(result)
                final_result.save(filename)

            except PySpin.SpinnakerException as ex:
                print('Error: %s' % ex)
                return False

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return result


def main():
    try:
        test_file = open('test.txt', 'w+')
    except IOError:
        print('Unable to write to the current directory. Please check permissions.')
        input('Press Enter to exit...')
        return False

    test_file.close()
    os.remove(test_file.name)

    result = True

    # Initialize the SLM device
    slm = slmdisplaysdk.SLMInstance()
    error = slm.open()
    assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

    # Initialize the camera
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    num_cameras = cam_list.GetSize()
    if num_cameras == 0:
        system.ReleaseInstance()
        slm.release()
        print('No cameras detected!')
        input('Done! Press Enter to exit...')
        return False

    cam = cam_list.GetByIndex(0)
    cam.Init()

    nodemap_tldevice = cam.GetTLDeviceNodeMap()
    nodemap = cam.GetNodeMap()

    # Start image acquisition and SLM control
    result &= acquire_images(cam, nodemap, nodemap_tldevice, slm)

    # Release camera and SLM resources
    cam.DeInit()
    del cam
    cam_list.Clear()
    system.ReleaseInstance()
    slm.release()

    input('Done! Press Enter to exit...')

    return result


if __name__ == '__main__':
    main()