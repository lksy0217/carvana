First, need to download carvana dataset.
Please refer to the **issue tab** for storage structure after downloading.

Next, need to make npz file.
      
      python data.py

if you want to train, Enter the command below.

      python run.py
      
if you want to test, Enter the command below.

      python run.py --mode test --test_cnt <int>

If you want to know the options to do this, Enter the command below.

      python run.py --help
      
      
If you want to check your study and test results, take a look at the **Result** directory

IDE used **vi** in **ubuntu 18.04** to **build the environment using anaconda**.

The **required libraries** are as follows.
      
      tensorflow-gpu 1.15
      keras
      opencv
      matplotlib
      PIL
      pandas
      numpy
      scikit-learn
