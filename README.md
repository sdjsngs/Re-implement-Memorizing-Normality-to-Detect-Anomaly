# Re-implement-Memorizing-Normality-to-Detect-Anomaly
Re-implement paper Memorizing Normality to Detect Anomaly
dataset
1 uscd-ped2 
  -training
    -frames
      -01
        -000.jpg
        -001.jpg
          .
          .
          .
       -02
        -000.jpg
          .
          .
          .
  -testing
    -same as training
  -ped2.mat
  
Avenue  same as uscd-ped2 
  -training
    -frames
      -01
        -000.jpg
        -001.jpg
          .
          .
          .
       -02
        -000.jpg
          .
          .
          .
  -testing
    -same as training
  -testing_label_mask
    -1_label.mat
    -2_label.mat
      .
      .
      .
 |Hyperparameter|              |
 | ---------- | :-----------:  |            
 |optimizer|sgd|
 |init lr|0.01|
 |lr schedule|step /10  after 10 epoch|
 |total epoch|40|
 

