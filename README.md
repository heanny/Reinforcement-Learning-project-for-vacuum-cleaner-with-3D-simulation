# Reinforcement-Learning-project-for-vacuum-cleaner-with-3D-simulation

Dear reader,

Welcome to our school cleaner agent testing stage! For running our project successully, we will kindly ask you to install the package with the correct version in the requirement.txt, and our python version is 3.9.12. Thanks! Here is our simulation envoriment (which is implemented by Unity) demo: https://youtu.be/etVIk2LVWXc

In our experimental design, we put the cleaner in four different environments to test and generated four figures to show the results. Thus in order to fully reproduce the experimental figures in our report, Please run the following jupyter notebook  respectively:

1. **Headless_room1.ipynb**
2. **Headless_room2.ipynb**
3. **Headless_room3.ipynb**
4. **Headless_room4.ipynb**

We intentionally set **smaller episode number** to make readers can plot and reproduce the results in a short time, so the performance of algorithms will suffer a loss because of that. The results figure is saved in the current directory with name **test\_results\_1.png**, **test\_results\_2.png**, **test\_results\_3.png** and **test\_results\_4.png**. 

The simulation environment of the agent is **a Unity application**, which interacts with our agent by executing the corresponding code in jupyter notebook (**NO** need to download anythings about unity). Different application packages are required on different operating systems, so we provided envoriment files for **Windows** and **MACOS** platform(sorry for Linux users!) In each notebook we will provide two file paths, which can be easily distinguished. These two paths are used to initialize the application pathname parameter **ENV_PATH**. You need to make sure that the path you choose to initialize **ENV_PATH** is corresponding to your operating system.

The unity interaction interface will pop up when you run the test, which uses the **port 5004**. Please follow the instructions in jupyter notebook if you would like to change it.

Have fun!

Best regards,

Group 6


## original version of unity model (only camera)
Windows 
https://drive.google.com/file/d/1Vvm10ijnrbAysdLdoKF7Gqs5zD3FysCG/view?usp=sharing

MACOS 
https://drive.google.com/file/d/1HGlqa6YWB3vmRz4GXi1Z3j0snhw9GeIm/view?usp=sharing


## final version of unity model (camera + ray sensor)
https://drive.google.com/file/d/1HGlqa6YWB3vmRz4GXi1Z3j0snhw9GeIm/view?usp=sharing
