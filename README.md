# easy-track

## state

- Now: under development

## Install

```
git clone https://github.com/LiYunfengLYF/easy-track
cd easy-track
python setup.py build
python setup.py install
```

## Document

### Python API (Directory)

test functions &nbsp;&nbsp;| [quick_start](#quick_start) | [run_sequence](#run_sequence) |
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|
[report_seq_performance](#report_seq_performance) |

board functions |

read functions&nbsp;&nbsp;| [imread](#imread) | [txtread](#txtread) | [seqread](#seqread) | 
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|
[img_filter](#img_filter) | [selectROI](#selectROI) | [img2tensor](#img2tensor) |

show functions | [imshow](#imshow) | [seqshow](#seqshow) | [close_cv2_window](#close_cv2_window) | [speed2waitkey](#speed2waitkey) |
<br> &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
&nbsp; | [show_otb](#show_otb) | [show_lasot](#show_lasot) | [show_uot](#show_uot) | [show_utb](#show_utb) |

draw functions &nbsp;| [draw_box](#draw_box) |

script functions | [remove_timetxt](#remove_timetxt) | [remove_same_img](#remove_same_img) | 
<br>&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp; | [trans_txt_delimiter](#trans_txt_delimiter) | [trans_imgs_order_name](#trans_imgs_order_name) |
<br>  &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp;| [extract_weights_from_checkpoint](#extract_weights_from_checkpoint) | [show_checkpoint_keys](#show_checkpoint_keys) |

trackers | [SiamFC](#SiamFC) | [Stark](#Stark) | [OSTrack](#OSTrack) | [LightFC](#LightFC) |

### test functions

__<span id="quick_start"> quick_start </span>__

```
def quick_start(tracker, seq_file, speed=20, imgs_type='.jpg'):
    """
    Description
        quick_start aim to help user to quickly observe the results of the tracker on an image sequence.
        It manually selects the initial bounding box and show results in each image by using blue bounding box.
    
    Params:
        tracker:
        seq_file:
        speed:      FPS speed, default = 20
        imgs_type:  image type, default = '.jpg'
    
    """
```

__<span id="run_sequence"> run_sequence </span>__

```
def run_sequence(tracker, seq_file, gt_file=None, save_path=None, save=False, visual=False, speed=20, imgs_type='.jpg',
                 select_roi=False, report_performance=True):
    """
    Description


    Params:
        tracker:    
        seq_file:   
        speed:      FPS speed, default = 20
        imgs_type:  image type, default = '.jpg'
    """
```

__<span id="report_seq_performance"> report_seq_performance </span>__

```
def report_seq_performance(gt_file, results_file):
    """
    Description
        calc Success Score, Precision Score, Norm Precision Score, Success Rate of on a sequence

    Params:
        gt_file:        str
        results_file:   str   

    """
```


### read functions

__<span id="imread"> imread </span>__

```
def imread():
    """
    Description
        imread is an extension of cv2.imread, which returns RGB images

    Params:
        filename:   the path of image
        
    Return:
        image:      np.array
    
    """
```

__<span id="txtread"> txtread </span>__

```
def txtread():
    """
    Description
        txtread is an extension of np.loadtxt, support ',' and '\t' delimiter.
        The original implementation method is in the pytracking library at https://github.com/visionml/pytracking

    Params:
        filename:           the path of txt
        delimiter:          default is [',','\t']
        dtype:              default is np.float64
    
    Return:
        ground_truth_rect:  np.array(n,4), n is length of results

    """
```

<span id="seqread"> seqread </span>

```
def seqread():
    """
    Description
        Seqread reads all image items in the file and sorts them by numerical name
        It returns a list containing the absolute addresses of the images

        Sorting only supports two types, '*/1.jpg' and '*/*_1.jpg'

    Params:
        file:       images' file
        imgs_type:  default is '.jpg'

    Return:
        List of absolute paths of sorted images

    """
```

__<span id="img_filter"> img_filter </span>__

```
def img_filter():
    """
    Description
        img_filter retains items in the specified format in the input list

    Params:
        imgs_list:          List of image path
        extension_filter:   default is '.jpg'

    Return:
        List of images path  with extension
        
    """
```

__<span id="selectROI"> selectROI </span>__

```
def selectROI(winname, img):
    """
    Description
        selectROI is an extension of cv2.selectROI
        input image is RGB rather BGR

    Params:
        winname:    name
        img:        np.array

    return:
        bbox:       [x,y,w,h]

    """
```

__<span id="img2tensor"> img2tensor </span>__

```
def img2tensor(img, device='cuda:0'):
    """
    Description
        transfer an img to a tensor
        mean: [0.485, 0.456, 0.406]
        std:  [0.229, 0.224, 0.225]

    Params:
        img:       np.array
        device:    default is 'cuda:0'

    return:
        Tensor:    torch.tensor(1,3,H,W)

    """
```

### show functions

__<span id="imshow"> imshow </span>__

```
def imshow():
    """
    Description
        imshow is an extension of cv2.imshow
        Different with cv2.imshow, it input is an RGB image, and window size is variable 

    Params:
        winname:    window name
        image:      np.array
        waitkey:    default is 0
    
    """
```

__<span id="seqshow"> seqshow </span>__

```
def seqshow():
    """
    Description
        seqshow visualizes the bounding box results of the tracker in image sequence

        if results_file is none, tracker results (default is red bounding box) will not be displayed on sequence
        if gt_file is none or show_gt is False, groundtruth (green bounding box) will not be displayed on sequence

    Params:
        imgs_file:      str
        imgs_type:      default is '.jpg', you can change it
        result_file:    str
        gt_file:        str
        show_gt:        True or False
        speed:          FPS
        tracker_name:   str
        seq_name:       str
        result_color:   default is red (0,0,255), you can change it
        thickness:      int
    
    """
```

__<span id="close_cv2_window"> close_cv2_window </span>__

```
def close_cv2_window(winname):
    """
    Description
        close an opened window of cv2

    Params:
        winname: str
        
    """
```

__<span id="speed2waitkey"> speed2waitkey </span>__

```
def speed2waitkey(speed):
    """
    Description
        trans fps to waitkey of cv2

    Params:
        speed:      fps, int

    return:
        waitket:    int
        
    """
```

__<span id="show_otb"> show_otb </span>__

```
def show_otb(dataset_files, result_file=None, speed=20, result_color=(0, 0, 255), show_gt=True, tracker_name=r''):
    """
    Description
        show_otb visualizes the bounding box results of the tracker in otb benchmark

        if results_file is none, tracker results (default is red bounding box) will not be displayed on OTB
        if show_gt is False, groundtruth (green bounding box) will not be displayed on OTB

    Params:
        dataset_files:  str
        result_file:    str
        speed:          FPS, default is 20
        result_color:   default is red (0,0,255), you can change it
        show_gt:        True or False
        tracker_name:   str
        
    """
```

__<span id="show_lasot"> show_lasot </span>__

```
def show_lasot(dataset_files, result_file=None, speed=20, result_color=(0, 0, 255), show_gt=True, tracker_name=r''):
    """
    Description
        show_lasot visualizes the bounding box results of the tracker in lasot benchmark

        if results_file is none, tracker results (default is red bounding box) will not be displayed on lasot
        if show_gt is False, groundtruth (green bounding box) will not be displayed on lasot

    Params:
        dataset_files:  str
        result_file:    str
        speed:          FPS, default is 20
        result_color:   default is red (0,0,255), you can change it
        show_gt:        True or False
        tracker_name:   str
        
    """
```

__<span id="show_uot"> show_uot </span>__

```
def show_uot(dataset_files, result_file=None, speed=20, result_color=(0, 0, 255), show_gt=True, tracker_name=r''):
    """
    Description
        show_uot visualizes the bounding box results of the tracker in uot benchmark

        if results_file is none, tracker results (default is red bounding box) will not be displayed on uot
        if show_gt is False, groundtruth (green bounding box) will not be displayed on uot

    Params:
        dataset_files:  str
        result_file:    str
        speed:          FPS, default is 20
        result_color:   default is red (0,0,255), you can change it
        show_gt:        True or False
        tracker_name:   str
        
    """
```

__<span id="show_utb"> show_utb </span>__

```
def show_utb(dataset_files, result_file=None, speed=20, result_color=(0, 0, 255), show_gt=True, tracker_name=r''):
    """
    Description
        show_utb visualizes the bounding box results of the tracker in uot benchmark

        if results_file is none, tracker results (default is red bounding box) will not be displayed on utb
        if show_gt is False, groundtruth (green bounding box) will not be displayed on utb

    Params:
        dataset_files:  str
        result_file:    str
        speed:          FPS, default is 20
        result_color:   default is red (0,0,255), you can change it
        show_gt:        True or False
        tracker_name:   str
        
    """
```

### draw functions

__<span id="draw_box"> draw_box </span>__

```
def draw_box(image, box, color, thickness):
    """
    Description
        draw a bounding box on image
    
    Params:
        image:      np.array
        box:        [x, y, w, h]
        color:      bounding box color
        thickness:  bounding box thickness
    
    Return:
        image:      np.array
    
    """
```

### script functions

__<span id="remove_timetxt"> remove_timetxt </span>__

```
def remove_timetxt(results_file):
    """
    Description
        Remove *_time.txt files from results_file

    Params:
        results_file:   file path
    
    """
```

__<span id="remove_same_img"> remove_same_img </span>__

```
def remove_same_img(file, save_file, checkpoint_path=None, device='cuda:0', resize=(320, 640), thred=0.4, show_same=False):
    """
    Description
        Remove same images in file and sort and save the rest images in save file
        It resizes input image to (320,640)(default) and uses MobileNetV2 to extract feature, then calc the similarity
        You need to sign the checkpoint_path of mobilenet_v2-b0353104.pth (from torchvision) and thred (default is 0.4)
        if not sign checkpoint path, it will search for weights in the etrack_checkpoints directory of the running .py file
        show_same=True will show the same image pair
    
    Params:
        results_file:       file path
        save_file:          file path
        checkpoint_path:    checkpoint path

    """
```

__<span id="trans_txt_delimiter"> trans_txt_delimiter </span>__

```
def trans_txt_delimiter(txt_file, out_file=None, delimiter=',', new_delimiter=None, with_time=True):
    """
    Description
        trans the delimiter of txt and save txt in out_file
        if out_file is none, out_file = txt_file
        if new_delimiter is none, new_delimiter = delimiter
        if with time is True, copy *_time.txt to out_file
        
    Params:
        txt_file:   source txt file
        out_file:   save txt file
        format:     ',' or '\t'
        new_format: ',' or '\t'
        with_time:  True or False

    """
```

__<span id="trans_imgs_order_name"> trans_imgs_order_name </span>__

```
def trans_imgs_order_name(file, save_file, sort=True, imgs_format='.jpg', preread=True, format_name=False, width=4,
                    start=1, end=None, ):
    """
    Description
        transfer image into an order name, and save in save_file
        if sort is False, it will directly read images. The original order of imgs may not be preserved
        if preread is False, it will directly copy and paste, image will not be opened
        if format_name is True, images' name is like 0001.jpg, 0002.jpg (width=4), ... else 1.jpg, 2.jpg, ...
    
    Params:
        file:           str
        save_file:      str
        sort:           True or False
        imgs_format:    str, default is '.jpg'
        preread:        True or False
        format_name:    True or False
        width:          int
        start:          int 
        end:            int 

    """
```

__<span id="extract_weights_from_checkpoint"> extract_weights_from_checkpoint </span>__

```
def extract_weights_from_checkpoint(checkpoint_file, out_file=None, name=None, key=None):
    """
    Description
        extract model's weight from checkpoint
        if out_file is None, save weights to current file
        if name is None, use current checkpoint name
        if key is None, use default names: 'net', 'network', 'model'

    Params:
        checkpoint_file:    checkpoint file
        out_file:           save weight file
        name:               save name, str type
        key:                key name, str type
    
    """
```

__<span id="show_checkpoint_keys"> show_checkpoint_keys </span>__

```
def show_checkpoint_keys():
    """
    Description
        show checkpoint keys

    Params:
        checkpoint_file:    checkpoint file

    """
```

### trackers

__How to obtain checkpoints for tracker in etrack-toolkit?__
<br>&nbsp;&nbsp;&nbsp;&nbsp;
go to tracker's official link, download the checkpoints
<br>&nbsp;&nbsp;&nbsp;&nbsp;
then use __extract_weights_from_checkpoint__ to extract the weight and rename it at the same time
<br>&nbsp;&nbsp;&nbsp;&nbsp;
then the new checkpoint can be load in tracker directly

- __<span id="SiamFC"> SiamFC </span>__

``` 
Tracker:  SiamFC
  Paper:  Fully-Convolutional Siamese Networks for Object Tracking
   Code:  https://github.com/huanglianghua/siamfc-pytorch     
```

- __<span id="Stark"> Stark </span>__

``` 
Tracker:  Starks50, StarkST50, StarkST101
  Paper:  Learning Spatio-Temporal Transformer for Visual Tracking    
   Code:  https://github.com/researchmm/Stark     
```

- __<span id="OSTrack"> OSTrack </span>__

``` 
Tracker:  OSTrack256, OSTrack384
  Paper:  Joint Feature Learning and Relation Modeling for Tracking: A One-Stream Framework    
   Code:  https://github.com/botaoye/OSTrack     
```

- __<span id="LightFC"> LightFC </span>__

``` 
Tracker:  LightFC
  Paper:  Lightweight Full-Convolutional Siamese Tracker   
   Code:  https://github.com/LiYunfengLYF/LightFC     
```

