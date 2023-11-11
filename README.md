# Automated Silicon-substrate Ultra-microtome (ASUM) 
ASUM is a novel mechanism based on circular silicon substrates for automatic collection of brain sections
<img src="./show_pics/ASUM_PIC.png" width="600px">


This repository contains the comprehensive release files of the ASUM:
- Mechanical design drawings of the ASUM
- Detection and control code of the ASUM 
- GUI of the ASUM

----

## Mechanical design drawings of the ASUM
<!-- ![alt](./show_pics/ASUM_mecha.png) -->
<img src="./show_pics/ASUM_mecha.png" width="400px">

- Open the design drawings by [UG](https://www.plm.automation.siemens.com/global/en/products/nx/) or [Solidworks](https://www.solidworks.com/zh-hans)
```
./Mechanical_drawing/Assembly_source_files-overall/装配体2(1).stp
```



----

## Detection and control code of the ASUM (Include GUI)
<img src="./show_pics/GUI2.png" width="400px">

**Note**: In this section, we assume that you are always in the directory $PROJECT_ROOT/

### Compatability
Currently this repo is compatible with ubuntu16.04/ubuntu18.04, Python 3, PyQt5.

### Running tutorial
#### 1. Environment configuration (for ubuntu OS)
```
pip install -r requirements.txt
```

#### 2. Download Pre-trained Model
  - Download address: [Pre-trained SSD detection model for brain sections](https://drive.google.com/file/d/1bxM01SwDm1i7HxVM0AG3kzNZ3eUrKP-M/view?usp=sharing)
  - Put the pre-trained model as `./Detection_and_control/ASUM-GUI-with-detection-and-control/models/SSD_sections_det.pth`

#### 3. Make sure the ASUM is installed successfully on the Ultra-microtome
  - make sure the serial port communication authority of the host is provided
  - make sure the electric motor has been successfully connected to computer
  - make sure that the CCD camera has been successfully connected to computer

#### 4. Run GUI 

(Normal mode: run with normal mode when there is an installed ASUM device)
```
python3 ./Detection_and_control/ASUM-GUI-with-detection-and-control/main-asum-gui-final.py
```
(Debug mode: run with normal mode when there is no existing ASUM device, only the GUI will be displayed by default under the debug mode)
```
python3 ./Detection_and_control/ASUM-GUI-with-detection-and-control/main-asum-gui-final.py --debug
```

----
## Declare
The Mechanical design drawings, GUI and the automatic control system of the ASUM can be used only for the purpose of academic study

## Citation
```
@article{cheng2021automated,
  title={Automated silicon-substrate ultra-microtome for automating the collection of brain sections in array tomography},
  author={Cheng, Long and Liu, Weizhou and Zhou, Chao and Zou, Yongxiang and Hou, Zeng-Guang},
  journal={IEEE/CAA Journal of Automatica Sinica},
  volume={8},
  number={2},
  pages={389--401},
  year={2021},
  publisher={IEEE}
}
```

