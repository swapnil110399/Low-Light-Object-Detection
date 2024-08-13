## ECE 176 Project Code
**In order to run the code**
1) To train the models make a dataset folder with folder structure
    Folder
        train
            high/*.jpg
            low/*.jpg
        test
            high/*.jpg
            low/*.jpg
    
    Train individual networks, 'decomposition_trainer.ipynb','restoration_trainer.ipynb' & 'illuminationAdjustment_trainer.ipynb' in the specific order. This will genearte a '.pth' files for each module, put them in Saved_models folder.

3) Finally, in the evaluate_model.ipynb file, make sure to have the test images in the specific location and adjust its path to the dataloader. Never mind the train path as we wont be using train data here, but make sure to have images in train as well to get the dataloader working. Then run the file, this will generate all the enhanced files into a folder.
4) Finally, install YOLO from git clone https://github.com/ultralytics/yolov5.git, 
then cd yolov5, 
then pip install -r requirements.txt, 
then wget https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt, 
then to get the YOLO Predictions **python detect.py --weights yolov5s.pt --img 256 --conf 0.25 --source Enhanced_images/** make sure to have correct path to detect.py (inside downloaded folder of yolo) and Enhanced_images folder (which contain enhanced images).
5) The predicted images are present in './yolov5/runs/detect/exp' folder. 

