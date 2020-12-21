# breast_histopathology 
Invasive Ductal Carcinoma is one of the most frequent types of breast cancer.   
The Invasive Ductal Carcinoma develops initially in milk duct's. These paths are responsible for transport the milk, which is produced in the lobes, to the nipple. As the cancer grows it tends to invade other areas of the breast, usually its tissues and lobes.  
The cancer cells are gradually surrounding the entire duct area and in last phase, the cells end to invade the breast tissues.

## Data
The Breast Histopathology presents a binary classification problem composed by 277524 samples.  
The samples correspond to scanned images of breast tissues, regarding several patients. The objective of the problem includes the correct classification of tissues with IDC and without IDC.  
The samples are available in RGB format and have dimensions of 50*50 pixels, respectively width and length.

## Limitations
The main limitations of this benchmark are:
* Moderate unbalanced classes (The distribution of the samples of the two classes shows a disproportion of 2.5-1 samples, respectively Without IDC and With IDC);
* Problem with high complexity. Although there is a large number of samples, the distinction between the two classes is not trivial;
* The models tend to be tedious, that is, they are much more likely to correctly classify samples from the most balanced class, thus resulting in very different performances between the two classes.
