import cv2

def calc_baseline_acc(generator, dataset_path, data_set_type):
    correct = 0.0
    for i in range(0, generator.n):
        label = generator.classes[i]
        
        p = "%s/%s/%s" % (dataset_path, 
                          data_set_type, 
                          generator.filenames[i])
        
        img = cv2.imread(p)
        im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(im, 
                                   cv2.HOUGH_GRADIENT,
                                   dp=2, 
                                   minDist=15, 
                                   param1=100, 
                                   param2=70)
        if circles is not None:
            if label == 1:
                correct += 1
        elif circles is None:
            if label == 0:
                correct += 1
    accuracy = (correct / generator.n)
    return {"acc": accuracy}

