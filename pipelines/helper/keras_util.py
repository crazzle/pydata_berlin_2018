import pickle


def build_generator(config_path, 
                    dataset_path, 
                    data_set_type,
                    img_height=100, img_width=100):
    
    with open(config_path, "r") as f:
        generator = pickle.load(f)
    
    path = "%s/%s" % (dataset_path, data_set_type)
    
    return generator.flow_from_directory(path,
                                         target_size=(img_height, 
                                                      img_width),
                                         color_mode='rgb')


